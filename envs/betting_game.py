import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class BettingGame(gym.Env):
    def __init__(self, win_prob, max_sequence_length, max_state_value=256, inital_state=16, reward_normalisation=256):        
        self.win_prob = win_prob
        self.max_sequence_length = max_sequence_length
        self.max_state_value = max_state_value
        self.initial_state = inital_state

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=self.max_state_value, shape=(2,), dtype=np.float32)
        self.reward_normalisation = reward_normalisation
        
        # Initialize state variables
        self.state = self.initial_state
        self.time = 0

    def reset(self):
        self.state = self.initial_state
        self.time = 0

        return self._get_observation(), {"mask" : np.ones(self.action_space.n, dtype=np.int8)}

    def step(self, action):
        bet = np.minimum(self.state * action / 8, self.max_state_value - self.state)

        did_win = random.random() < self.win_prob
        reward = (2 * did_win - 1) * bet
        self.state = self.state + reward
        self.time += 1
        
        done = self.time >= self.max_sequence_length or self.state == self.max_state_value or self.state == 0

        # Return the observation, reward, done, and any additional info
        return self._get_observation(), reward / self.reward_normalisation, done, {"mask" : np.ones(self.action_space.n, dtype=np.int8)}

    def _get_observation(self):
        return np.array([self.state / 256, self.time / 6], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Current state: {self.state}")

    def close(self):
        pass
