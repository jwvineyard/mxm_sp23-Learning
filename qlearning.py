from collections import defaultdict
import numpy as np
from daw_etal_two_stage import DawEtalTwoStageMDP
from agent import Agent, train_episode, train, rate_decay
from itertools import islice

'''
A Q-Learning implementation for a Gymnasium environment (with a discrete observation/action space).
'''
class QLearning(Agent):
    def __init__(self,
            environment,
            discount_rate=0.95,
            learning_rate=rate_decay(),
            exploration_rate=rate_decay(base=0.1),
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
    
    def q_update(self,
            observation,
            action,
            reward,
            terminated,
            truncated,
            next_observation):
        future_q_value = (not (terminated or truncated)) * np.max(self.q_values[next_observation])
        temporal_difference = (
            reward + self.discount_rate * future_q_value - self.q_values[observation][action]
        )
        self.q_values[observation][action] = (
            self.q_values[observation][action] + self.learning_rate * temporal_difference
        )
        return temporal_difference

    def update(self,
            observation,
            action,
            reward,
            terminated,
            truncated,
            next_observation):
        temporal_difference = self.q_update(observation, action, reward, 
                                            terminated, truncated, next_observation)
        self.training_error.append(temporal_difference)
        if self.t % 1000 == 0:
            print(np.mean(list(islice(reversed(self.training_error), 10000))), self.learning_rate, self.exploration_rate)

class DynaQ(QLearning):
    def __init__(self,
            environment,
            discount_rate=0.95,
            learning_rate=rate_decay(),
            exploration_rate=rate_decay(),
            n_simulations=2000,
            seed=None):
        super().__init__(environment, discount_rate, learning_rate, exploration_rate, seed)
        self.n_simulations = n_simulations
        # Dictionary where keys are current states, and values are arrays the size of the action space, 
        # where for each action we have a pair of (reward, next_state).
        # self.model = defaultdict(lambda: np.zeros(self.env.action_space.n, dtype=[('reward', np.float32), ('next_state', np.intc)]))
        self.model = dict()

    def sample_state(self):
        return self.rng.choice(list(map(lambda x: x[0], self.model.keys())))
    
    def sample_action(self, state):
        return self.rng.choice(list(map(lambda x: x[1], filter(lambda x: x[0] == state, self.model.keys()))))

    def update(self,
            observation,
            action,
            reward,
            terminated,
            truncated,
            next_observation):
        
        # Q-learning update
        temporal_difference = self.q_update(observation, action, reward, 
                                            terminated, truncated, next_observation)
        self.training_error.append(temporal_difference)
        
        # Update model (w/ assumption of deterministic environment, see Sutton and Barto)
        self.model[(observation, action)] = (reward, next_observation)

        # simulated Q-learning
        for i in range(self.n_simulations):
            s = self.sample_state()
            a = self.sample_action(s)
            (r, s_) = self.model[(s,a)]
            self.q_update(s, a, r, False, False, s_)

        if self.t % 10 == 0:
            print(np.mean(list(islice(reversed(self.training_error), 10000))), self.learning_rate, self.exploration_rate)

def main():
    env = DawEtalTwoStageMDP(episodic=True)
    dyna = DynaQ(env)
    train_episode(dyna, env)
    # print(dyna.sample_action(dyna.sample_state()))

    train(dyna, env, n_episodes=2_000)

if __name__ == "__main__":
    main()
