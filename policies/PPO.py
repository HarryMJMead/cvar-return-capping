import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, MaskedCategorical
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

from policies.base_policy import BasePolicy, get_observation_size


class PolicyNet(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size=128):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.layer3(x)


class PPO(BasePolicy):

    def __init__(
        self,
        gamma, 
        lr, 
        env,
        epochs_per_batch,
        sub_batch_size,
        frames_per_batch,
        alpha_sub_batch,
        alpha_batch,
        clip_eps,
        entropy_eps, 
        lmbda,
        max_grad_norm,
        seed,
        hidden_size,
        device
    ):
        super().__init__(
            gamma=gamma, 
            lr=lr, 
            env=env,
            epochs_per_batch=epochs_per_batch,
            sub_batch_size=sub_batch_size,
            frames_per_batch=frames_per_batch,
            alpha_sub_batch=alpha_sub_batch,
            alpha_batch=alpha_batch,
            seed=seed,
            device=device
        )
        self.clip_eps = clip_eps
        self.entropy_eps = entropy_eps
        self.lmbda = lmbda
        self.max_grad_norm = max_grad_norm

        torch.manual_seed(self.seed)
        observation_size = get_observation_size(self.env.observation_space)
        print(observation_size)
        self.policy_net = PolicyNet(observation_size, self.env.action_space.n, hidden_size=hidden_size).to(device)
        self.value_net = PolicyNet(observation_size, 1, hidden_size=hidden_size).to(device)
        self.value_module = ValueOperator(
            module=self.value_net,
            in_keys=['observation']
        )

        self.policy_module = TensorDictModule(
            self.policy_net, in_keys=["observation"], out_keys=["logits"]
        )

        self.policy_module = ProbabilisticActor(
            module=self.policy_module,
            in_keys=["logits", 'mask'],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
            default_interaction_type=InteractionType.RANDOM,
            # we'll need the log-prob for the numerator of the importance weights
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_eps,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), self.lr)

        self.advantage_module = GAE(
            gamma=self.gamma, lmbda=self.lmbda, value_network=self.value_module, average_gae=True
        )
    

    def select_action(self, state : torch.Tensor, mask : torch.Tensor, deterministic=False):
        if not deterministic:
            with torch.no_grad():
                td_data = TensorDict({'observation': state, 'mask': mask}, [1,])
                self.policy_module(td_data)
                return td_data['action'], td_data['sample_log_prob']
        else:
            with torch.no_grad():
                action_vals = self.policy_net(state)
                min_mask = (mask.type(torch.float32) - 0.5) * np.inf

                action_vals = torch.minimum(action_vals, min_mask)
                probs = F.softmax(action_vals, dim=1)
                
                action_prob, action = torch.max(probs, dim=1)
                return action, torch.log(action_prob)


    def preprocess_batch(self, tensordict_data: TensorDict):
        self.advantage_module(tensordict_data)

        
    def optimize_model(self, sub_batch : TensorDict):
        loss_vals = self.loss_module(sub_batch)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        
        if wandb.run is not None:
            wandb.log({'Policy/Entropy': loss_vals["entropy"].item()}, commit=False)

        # Optimization: backward, grad clipping and optimization step
        loss_value.backward()
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pt")
        torch.save(self.value_net.state_dict(), f"{path}/value_net.pt")
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(f"{path}/policy_net.pt"))
        self.value_net.load_state_dict(torch.load(f"{path}/value_net.pt"))
