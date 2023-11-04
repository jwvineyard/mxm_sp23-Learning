import numpy as np
import pandas as pd
from functools import reduce
from itertools import product

# The entirety of this code has been respectfully snagged from:
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

class MDP:
    def __init__(self, states=1, rewards=[-1,0,1], actions=lambda s: 2):
        self.states = states;
        self.rewards = rewards;
        self.actions = actions;

    def p(self, s, r, s_, a):
        n = self.states * len(self.rewards) * self.states * reduce(lambda a,b: a*b,
                                                              map(
                                                                  lambda s: self.actions(s),
                                                                  range(0, self.states)
                                                              ))
        return float(1)/float(n)

    def initial_state(self): return 0

    def next_state(self, s, a):
        l = list(map(lambda x: (x[0], x[1], self.p(s, x[0], x[1], a)), product(self.rewards, range(self.states))))
        assert(reduce(lambda x, y: x + y[2], l, 0) == 1)
        sample = np.random.rand()
        for (r, s_, x) in l:
            sample -= x
            if sample < 0:
                return (r, s_)

# States: 0 (Blue), 1 (Pink), 2 (Green)
# Actions: 0 (Left), 1 (Right)
class DawEtalTwoStageMDP(MDP):
    def __init__(self, rewards):
        super(DawEtalTwoStageMDP, self).__init__(3, rewards, actions=lambda s: 2)

    def p(self, s, r, s_, a):
        match [s,r,s_,a]:
            case [0, 0, 1, 0]: return 0.3
            case [0, 0, 1, 1]: return 0.7
            case [0, 0, 2, 0]: return 0.7
            case [0, 0, 2, 1]: return 0.3
            case [1, r, 0, 0]: return 1/len(self.rewards)
            case [1, r, 0, 1]: return 1/len(self.rewards)
            case [2, 0, 0, 0]: return 1
            case [2, 0, 0, 1]: return 1
            case _: return 0

class DiceGameMDP(MDP):
    def __init__(self):
        super(DiceGameMDP, self).__init__(states=2, rewards=[0,3,5], actions=lambda s: 2)


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

def main():
    mdp = DawEtalTwoStageMDP([0,1])
    print(mdp.p(0, 0, 2, 1))
    s = mdp.initial_state()
    rl = QLearningTable([0,1])
    for _ in range(0, 1000):
        a = rl.choose_action(s)
        (r, s_) = mdp.next_state(s, 0)
        rl.learn(s, a, r, s_)
        print((s,a,r,s_))
        s = s_
    print(rl.q_table)


if __name__ == "__main__":
    main()
