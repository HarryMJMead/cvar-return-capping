import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensordict import TensorDict

from policies.base_policy import BasePolicy


class PolicyNet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN(BasePolicy):

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
        eps_start,
        eps_end,
        eps_decay,
        tau,
        max_grad_value,
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
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.max_grad_value = max_grad_value

        self.policy_net = PolicyNet(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=256).to(device)
        self.target_net = PolicyNet(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=256).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        self.steps_done = 0

    
    def select_action(self, state: torch.Tensor, mask: torch.Tensor, deterministic=False):
        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        if not deterministic:
            self.steps_done += 1
        if sample > eps_threshold or deterministic:
            with torch.no_grad():
                min_mask = (mask.type(torch.float32) - 0.5) * np.inf

                q_vals = self.policy_net(state)
                q_vals = torch.minimum(q_vals, min_mask)

                return torch.argmax(q_vals, dim=1), None
        else:
            possible_actions = torch.arange(self.n_actions).unsqueeze(0)[mask.cpu()]
            action = np.random.choice(possible_actions)

            return torch.tensor([action], dtype=torch.int64, device=self.device), None


    def optimize_model(self, sub_batch: TensorDict):
        sub_batch.set('q_value', self.policy_net(sub_batch.get('observation')))
        sub_batch.set('action_value', sub_batch.get('q_value').gather(1, sub_batch.get('action').unsqueeze(1)))

        sub_batch.set(('next', 'state_value'), torch.zeros(self.sub_batch_size, device=self.device))

        non_final_next_states_mask = ~sub_batch.get(('next', 'done'))
        non_final_next_states = sub_batch.get(('next', 'observation'))[non_final_next_states_mask]
        non_final_next_states_action_mask = sub_batch.get(('next', 'mask'))[non_final_next_states_mask]

        with torch.no_grad():
            non_final_min_mask = (non_final_next_states_action_mask.type(torch.float32) - 0.5) * np.inf
            non_final_q_vals = torch.minimum(self.target_net(non_final_next_states), non_final_min_mask)
            sub_batch[('next', 'state_value')][non_final_next_states_mask] = non_final_q_vals.max(1).values

        sub_batch.set('value_target', (sub_batch.get(('next', 'state_value')) * self.gamma) + sub_batch.get(('next', 'reward')))

        criterion = nn.MSELoss()
        loss = criterion(sub_batch.get('action_value'), sub_batch.get('value_target').unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


