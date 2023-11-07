import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, seed=None):
        np.random.seed(seed)
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.correct_choices_phase1 = 0
        self.correct_choices_phase3 = 0
        self.total_choices_phase1 = 0
        self.total_choices_phase3 = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[state])
   
    def train(self, max_steps=50):
        phases = ['I', 'II', 'III']
        correct_ratio_phase1_list = []
        correct_ratio_phase3_list = []
        for phase in phases:
            self.env.set_phase(phase)
            state = self.env.reset()
            
            # Reset Q-values at the beginning of each phase
            self.q_values = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-values
                self.q_values[state, action] = (1 - self.alpha) * self.q_values[state, action] + \
                                               self.alpha * (reward + self.gamma * np.max(self.q_values[next_state]))
    
                # For calculating the correct choices to total choices ratio
                if phase == 'I' or phase == 'III':
                    correct_action = np.argmax([4, 2]) if state == 1 else np.argmax([2, 4])
                    if action == correct_action:
                        self.correct_choices_phase1 += 1 if phase == 'I' else 0
                        self.correct_choices_phase3 += 1 if phase == 'III' else 0
                    self.total_choices_phase1 += 1 if phase == 'I' else 0
                    self.total_choices_phase3 += 1 if phase == 'III' else 0
    
                    # Update and store the correct ratio for plotting
                    if phase == 'I' and self.total_choices_phase1 > 0:
                        correct_ratio_phase1_list.append(self.correct_choices_phase1 / self.total_choices_phase1)
                    if phase == 'III' and self.total_choices_phase3 > 0:
                        correct_ratio_phase3_list.append(self.correct_choices_phase3 / self.total_choices_phase3)
    
                state = next_state
    
            # Save the Q-values after Phase II
            if phase == 'II':
                phase_ii_q_values = self.q_values.copy()
    
        # Line plotting the correct ratio for Phase I and Phase III
        plt.plot(range(len(correct_ratio_phase1_list)), correct_ratio_phase1_list, label='Phase I')
        plt.plot(range(len(correct_ratio_phase3_list)), correct_ratio_phase3_list, label='Phase III')
        plt.ylabel('Correct Choices Ratio')
        plt.ylim(0, 1)
        plt.xlabel('Total Choices Made')
        plt.title('Performance of Q-Learning Agent')
        plt.legend()
        plt.show()
    
        # At the end of training, compare Q-values learned in Phase II with those in Phase III
        q_value_diff = np.abs(self.q_values - phase_ii_q_values)
        print("Difference in Q-values between Phase II and III:", np.sum(q_value_diff))
       
        return correct_ratio_phase1_list, correct_ratio_phase3_list

