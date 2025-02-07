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

class MixingNet(nn.Module):
    def __init__(self, n_observations, hidden_size=64):
        super(MixingNet, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class MixedPolicy(nn.Module):
    def __init__(self, pi_1, pi_2, m_net):
        super(MixedPolicy, self).__init__()
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.m_net = m_net

    def forward(self, x):
        m = self.m_net(x)  # Get mixing parameter
        logits_1 = self.pi_1(x)  # Logits from pi_1
        logits_2 = self.pi_2(x)  # Logits from pi_2

        # Convert logits to probabilities using softmax
        probs_1 = F.softmax(logits_1, dim=-1)
        probs_2 = F.softmax(logits_2, dim=-1)

        # Mix the probabilities
        mixed_probs = m * probs_1 + (1 - m) * probs_2

        # Convert back to logits using log (to be used in categorical distribution)
        mixed_logits = torch.log(mixed_probs + 1e-8)  # Small epsilon to prevent log(0)

        return mixed_logits


class MIX(BasePolicy):

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
        risk_neutral_path,
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

        self.mixing_net = MixingNet(self.env.observation_space.shape[0], hidden_size=hidden_size).to(device)
        self.policy_net = PolicyNet(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=hidden_size).to(device)
        self.risk_neutral_net = PolicyNet(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=hidden_size).to(device)
        self.risk_neutral_net.load_state_dict(torch.load(f"{risk_neutral_path}/policy_net.pt", map_location=device))

        # Make sure pi_2 is not updated during training
        for param in self.risk_neutral_net.parameters():
            param.requires_grad = False

        self.mixed_policy = MixedPolicy(self.policy_net, self.risk_neutral_net, self.mixing_net)

        self.policy_module = TensorDictModule(
            self.mixed_policy, in_keys=["observation"], out_keys=["logits"]
        )

        self.policy_module = ProbabilisticActor(
            module=self.policy_module,
            in_keys=["logits", 'mask'],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
            default_interaction_type=InteractionType.RANDOM,
            # we'll need the log-prob for the numerator of the importance weights
        )

        params_to_optimise = list(self.policy_net.parameters()) + list(self.mixing_net.parameters())
        self.optimizer = torch.optim.Adam(params_to_optimise, self.lr)
    

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
        log_prob = dist.log_prob(sub_batch['action'])

        VaR_relative_return = (sub_batch['final_return'] - sub_batch['VaR']) 
        loss_value = - torch.mean(VaR_relative_return * log_prob)

        # Optimization: backward, grad clipping and optimization step
        loss_value.backward()
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        torch.nn.utils.clip_grad_norm_(self.policy_module.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pt")
            
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(f"{path}/policy_net.pt"))