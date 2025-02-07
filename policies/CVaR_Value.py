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

from policies.base_policy import BasePolicy


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


class CVaR_Value(BasePolicy):

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
        max_grad_norm,
        hidden_size,
        seed,
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
        self.max_grad_norm = max_grad_norm

        self.policy_net = PolicyNet(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=hidden_size).to(device)
        self.value_net = PolicyNet(self.env.observation_space.shape[0], 1, hidden_size=hidden_size).to(device)
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

        self.optimizer = torch.optim.Adam(self.policy_module.parameters(), self.lr)
        self.value_optimizer = torch.optim.Adam(self.value_module.parameters(), self.lr)
    

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

        
    def optimize_model(self, sub_batch : TensorDict):

        dist = self.policy_module.get_dist(sub_batch)
        self.value_module(sub_batch)
        log_prob = dist.log_prob(sub_batch['action'])

        VaR_relative_return = (sub_batch['final_return'] - sub_batch['VaR'])
        with torch.no_grad(): 
            advantage = VaR_relative_return - sub_batch['state_value']
        policy_loss = -torch.mean(advantage * log_prob)

        # Optimization: backward, grad clipping and optimization step
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_module.parameters(), self.max_grad_norm)
        self.optimizer.step()

        value_loss = torch.mean((VaR_relative_return - sub_batch['state_value'])**2)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_module.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pt")
            
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(f"{path}/policy_net.pt"))