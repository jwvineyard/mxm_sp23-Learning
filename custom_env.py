import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class CustomEnv(gym.Env):
    def __init__(self,seed=None):
        np.random.seed(seed)
        # Action Space: 0 for 'left', 1 for 'right'
        self.action_space = spaces.Discrete(2)
        
        # State Space: 'S0', 'S1', 'S2'
        self.observation_space = spaces.Discrete(3)

        self.state = 0  # Start state
        self.phase = "I"  # Start Phase

    def _get_transition_probabilities(self, state, action):
        if action == 0:  # Left action
            return [(1, 0.7), (2, 0.3)]
        else:  # Right action
            return [(1, 0.3), (2, 0.7)]
            
    def step(self, action):
        done = False
        reward = 0
        info = {}

        if self.phase == "I":
            # Phase I
            self.state = self._transition(action)
            reward = 4 if self.state == 1 else 2
            
        elif self.phase == "II":
            # Phase II: Agent trapped in event horizon (S1)
            self.state = 1
            reward = 1

        else:
            # Phase III
            self.state = self._transition(action)
            reward = 1 if self.state == 1 else 2  # Rewards now as per Phase II
            
        return self.state, reward, done, info

    def reset(self):
        self.state = 0
        return self.state

    def _transition(self, action):
        if action == 0:  # Left action
            return np.random.choice([1, 2], p=[0.7, 0.3])
        else:  # Right action
            return np.random.choice([1, 2], p=[0.3, 0.7])

    def set_phase(self, phase):
        self.phase = phase

