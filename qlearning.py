from collections import defaultdict
import numpy as np
from daw_etal_two_stage import DawEtalTwoStageMDP
from agent import Agent, train_episode, train

'''
A Q-Learning implementation for a Gymnasium environment (with a discrete observation/action space).
'''
class QLearning(Agent):
    def rate_decay(base=0.95, decay=0.995):
        return lambda t: base * (decay ** t)

    def __init__(self,
            environment,
            discount_rate=0.95,
            learning_rate=rate_decay(),
            exploration_rate=rate_decay(),
            seed=None):
        self.env = environment
        # self.t = 0
        self.training_error = []
        self.rng = np.random.default_rng(seed)

        self.discount_rate = discount_rate
        self._learning_rate = learning_rate
        self._exploration_rate = exploration_rate

        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
    
    @property
    def t(self):
        return len(self.training_error)

    @property
    def learning_rate(self):
        return self._learning_rate(self.t)

    @property
    def exploration_rate(self):
        return self._exploration_rate(self.t)

    def get_action(self, observation):
        if self.rng.random() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[observation]))
    
    def update(self,
            observation,
            action,
            reward,
            terminated,
            truncated,
            next_observation):
        future_q_value = (not terminated) * np.max(self.q_values[next_observation])
        temporal_difference = (
            reward + self.discount_rate * future_q_value - self.q_values[observation][action]
        )
        self.q_values[observation][action] = (
            self.q_values[observation][action] + self.learning_rate * temporal_difference
        )
        self.training_error.append(temporal_difference)

def main():
    env = DawEtalTwoStageMDP(episodic=True)
    train(QLearning(env), env, n_episodes=200_000)

if __name__ == "__main__":
    main()
