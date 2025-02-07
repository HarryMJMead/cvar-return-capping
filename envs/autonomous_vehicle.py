import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt
import random

def load_edge(map_path):
    return np.load(map_path)

class AutonomousVehicle(gym.Env):
    def __init__(self, map_path, edge_rewards, edge_probs, max_steps, one_hot=False, reward_normalisation=256) -> None:
        self.one_hot = one_hot
        self.edge_rewards = np.array(edge_rewards, dtype=np.float32)
        self.edge_probs = np.array(edge_probs, dtype=np.float32)

        self.max_steps = max_steps
        map_edges = np.load(map_path)
        self.map_edges_horizontal, self.map_edges_vertical = map_edges["horizontal"], map_edges["vertical"]
        self.map_list = [self.map_edges_vertical, self.map_edges_horizontal]
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        self.max_x = self.map_edges_vertical.shape[1]
        self.max_y = self.map_edges_horizontal.shape[0]

        self.reward_normalisation = reward_normalisation

        self.actions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        self.action_space = spaces.Discrete(4)
        if self.one_hot:
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_x * self.max_y + 1,), dtype=np.float32)  
        else:
            self.observation_space = spaces.Box(low=0, high=np.array([self.max_x-1, self.max_y-1, self.max_steps]), shape=(3,), dtype=np.float32)

    def reset(self):
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        return self.state if not self.one_hot else self._get_embedded_state(), {"mask" : self._get_mask(self.state)}
    
    def _get_mask(self, state):
        mask = np.array([
            (state[1] != self.max_y-1),
            (state[0] != self.max_x-1),
            (state[1] != 0),
            (state[0] != 0)
        ])

        return mask.astype(np.int8)
    
    def step(self, action):
        state_change = self.actions[action]

        direction = action // 2
        axis = action % 2

        path_value = self.map_list[axis][self.state[1].astype(np.int8) + (1-axis)*(-direction), self.state[0].astype(np.int8) + axis*(-direction)]

        self.state[0:2] += state_change
        self.step_count += 1

        rand_choice = random.random()
        result = 1 + (rand_choice > self.edge_probs[1]) - (rand_choice < self.edge_probs[0])

        reward = -self.edge_rewards[path_value, result]

        done = (self.state[0:2] == np.array([self.max_x-1, self.max_y-1])).all()

        reward += 80 * done
        self.state[2] -= reward/64

        done = np.logical_or(done, self.step_count >= self.max_steps)

        return self.state if not self.one_hot else self._get_embedded_state(), reward / self.reward_normalisation, done, {"mask" : self._get_mask(self.state)}
    
    def _get_embedded_state(self):
        embedded_state = np.zeros((self.max_x*self.max_y + 1), dtype=np.float32)
        embedded_state[-1] = self.state[-1]
        embedding_indices = self.state[0]*self.max_y + self.state[1]
        embedded_state[embedding_indices.astype(np.int32)] = 1

        return embedded_state

