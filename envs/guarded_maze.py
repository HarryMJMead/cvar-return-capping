import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt

class GuardedMaze(gym.Env):
    def __init__(self, start = [6, 1], goal = [5, 6], guard = [6, 5], max_steps = 500, deduction_limit = 32, reward_normalisation=256) -> None:
        self.max_steps = max_steps
        self.deduction_limit = deduction_limit
        self.map = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, -1, -1, -1, -1, 0, -1],
                    [-1, 0, 0, 0, 0, -1, 0, -1],
                    [-1, 0, 0, 0, 0, -1, 0, -1],
                    [-1, 0, 0, 0, 0, -1, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1]])

        if start == 'random':
            self.start = self._get_random_start()
            self.random_start = True
        else:
            self.start = tuple(start)
            self.random_start = False

        self.goal = tuple(goal)
        self.guard = tuple(guard)

        self.map[self.start] = 1
        self.map[self.goal] = 2
        self.map[self.guard] = 3

        self.actions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        self.state = np.array(self.start, dtype=np.float32)
        self.mask = self._get_mask(self.state)
        self.steps = 0
        self.tot_reward = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(64+1,), dtype=np.float32)
        self.reward_normalisation = reward_normalisation

    def show_state(self):
        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(self.map)
        ax.add_patch(plt.Circle(np.flip(self.state, axis=0), 0.3, color='r'))
    
    def render(self, mode='human'):
        self.show_state()
    
    def _get_mask(self, state):
        mask = [self.map[(tuple(state.astype(np.int64) + action))] != -1 for action in self.actions]
        return np.array(mask).astype(np.int8)
    
    def _get_random_start(self):
        start_type = -1
        while start_type != 0:
            start = np.random.randint(0, 8, size=(2,))
            start_type = self.map[(tuple(start))]
        return tuple(start)
    
    def reset(self):
        if self.random_start:
            self.map[self.start] = 0
            self.start = self._get_random_start()
            self.map[self.start] = 1
        self.state = np.array(self.start, dtype=np.float32)
        self.mask = self._get_mask(self.state)
        self.steps = 0
        self.tot_reward = 0

        return self.get_augmented_obs(self._get_encoded_state()), {"mask" : self.mask.copy()}
    
    def step(self, action):
        self.state += self.actions[action]
        self.mask = self._get_mask(self.state)
        self.steps += 1

        reward = 0
        if self.steps <= self.deduction_limit:
            reward -= 1

        if self.map[tuple(self.state.astype(np.int32))] == 3:
            reward += np.random.normal()*30
        
        done = self.map[tuple(self.state.astype(np.int32))].item() == 2
        reward += 10 * done

        self.tot_reward += reward

        if self.steps >= self.max_steps:
            done = True

        return self.get_augmented_obs(self._get_encoded_state()), reward / self.reward_normalisation, done, {"mask" : self.mask.copy()}

    def _get_encoded_state(self):
        state_encoded = np.zeros(64, dtype=np.float32)
        state_encoded[int(self.state[0]*8 + self.state[1])] = 1
        return state_encoded
    
    def get_augmented_obs(self, obs):
        return np.append(obs, self.tot_reward / self.reward_normalisation).astype(np.float32)

class GuardedMazeNoMask(GuardedMaze):
    def reset(self):
        state, _ = super().reset()
        return state, np.ones(4, dtype=bool)

    def step(self, action, parameters):
        if self.mask[action]:
            _, reward, done, _ = super().step(action, parameters)
        else:
            self.steps += 1
            done = self.steps >= self.max_steps
            reward = -1/self.reward_normalisation if self.steps <= self.deduction_limit else 0
        
        return self._get_encoded_state(), reward, done, {"mask": np.ones(4, dtype=bool)}

    

