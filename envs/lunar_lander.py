import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class LunarLander(gym.Env):
    def __init__(self, render_mode="none", seed=4, reward_normalisation=256):
        if render_mode != "none":
            self.env = gym.make("LunarLander-v3", render_mode=render_mode)
        else:
            self.env = gym.make("LunarLander-v3")
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=np.append(self.env.observation_space.low, np.array(-512/reward_normalisation, dtype=np.float32)),
            high=np.append(self.env.observation_space.high, np.array(512/reward_normalisation, dtype=np.float32)),
            dtype=np.float32,
        )   
        self.seed = seed
        self.reward_normalisation = reward_normalisation
        self.total_reward = 0

    def reset(self):
        observation, info =  self.env.reset(seed=self.seed)
        self.total_reward = 0
        info["mask"] = np.ones(self.action_space.n, dtype=np.int8)

        return np.append(observation, np.array(self.total_reward, dtype=np.float32)), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["mask"] = np.ones(self.action_space.n, dtype=np.int8)

        if terminated and observation[0] > 0:
            reward_noise = 70 + np.random.normal(0, 1)*100
            reward += reward_noise
        
        normalised_reward = reward / self.reward_normalisation
        self.total_reward += normalised_reward

        # Return the observation, reward, done, and any additional info
        return np.append(observation, np.array(self.total_reward, dtype=np.float32)), normalised_reward, terminated or truncated, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()
