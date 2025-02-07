from abc import ABC, abstractmethod
import torch
import torch.optim as optim

from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage


class BasePolicy(ABC):
    optimizer : optim.Optimizer

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
        seed,
        device
    ):
        self.gamma = gamma
        self.lr = lr
        self.env = env
        self.epochs_per_batch = epochs_per_batch
        self.sub_batch_size = sub_batch_size
        self.frames_per_batch = frames_per_batch
        self.alpha_sub_batch = alpha_sub_batch
        self.alpha_batch = alpha_batch
        self.seed = seed
        self.device = device

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=int(self.frames_per_batch / self.alpha_sub_batch)),
            sampler=SamplerWithoutReplacement(),
        )
    
    @abstractmethod
    def select_action(self, state : torch.Tensor, mask : torch.Tensor, deterministic=False):
        ...

    @abstractmethod
    def optimize_model(self, sub_batch : TensorDict):
        ...

    @abstractmethod
    def save(self, path):
        ...
    
    @abstractmethod
    def load(self, path):
        ...
    
    def preprocess_batch(self, tensordict_data : TensorDict):
        pass

    def train(self, tensordict_data : TensorDict):
        for _ in range(self.epochs_per_batch):
            self.preprocess_batch(tensordict_data)
            data_view = tensordict_data.reshape(-1)

            if self.alpha_batch != 1:
                sorted_returns, idx = data_view['final_return'].sort() 
                data_view = data_view[idx[:int(data_view.shape[0] * self.alpha_batch)]]
                data_view['VaR'] = sorted_returns[int(sorted_returns.shape[0] * self.alpha_batch)] * torch.ones(data_view.shape[0], device=self.device)

            self.replay_buffer.empty()
            self.replay_buffer.extend(data_view.cpu())
            
            for i in range(self.frames_per_batch // self.sub_batch_size):
                sub_batch = self.replay_buffer.sample(int(self.sub_batch_size / self.alpha_sub_batch)).to(self.device)

                if self.alpha_sub_batch != 1:
                    _, idx = sub_batch['final_return'].sort() 
                    sub_batch = sub_batch[idx[:self.sub_batch_size]]

                self.optimize_model(sub_batch)
            
