import torch
from tensordict import TensorDict
from collections import namedtuple


Episode = namedtuple('Episode',
                        ('final_return', 'frames'))

class ReplayMemory():

    def __init__(
        self,
        frames_per_batch,
        device,
        alpha_batch = 1.0,
        alpha_sub_batch = 1.0,
    ) -> None:
        
        self.frames_per_batch = (frames_per_batch / alpha_batch) / alpha_sub_batch

        self.episode_rollout : list[TensorDict] = []
        self.episode_memory : list[Episode] = []

        self.device=device

        self.frames_in_memory = 0


    def append_frame(self, frame : TensorDict):
        self.episode_rollout.append(frame)
    

    def finish_episode(self, final_return):
        self.frames_in_memory += len(self.episode_rollout)

        rollout_tensor_dict = torch.cat(self.episode_rollout)
        rollout_tensor_dict.set('final_return', torch.ones(len(self.episode_rollout), device=self.device) * final_return)

        self.episode_memory.append(Episode(final_return, rollout_tensor_dict))
        self.episode_rollout : list[TensorDict] = []

        if self.frames_in_memory >= self.frames_per_batch:
            return True
        return False
    

    def get_tensordict(self):
        tensordict_data = torch.cat([ep.frames for ep in self.episode_memory])
        self.episode_memory : list[Episode] = []
        self.frames_in_memory = 0

        return tensordict_data
        