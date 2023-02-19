import gymnasium as gym
from gymnasium import spaces
import itertools
import operator

def unzip(i):
    l, r = itertools.tee(i);
    return (list(map(operator.itemgetter(0), l)), list(map(operator.itemgetter(1), r)))

'''
A [Gymnasium](https://gymnasium.farama.org/) environment for Daw, et. al. (2011) two stage Markov decision process.
'''
class DawEtalTwoStageMDP(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    probabilities = [[0.3, 0.7], [0.7, 0.3]]

    def __init__(self, 
            render_mode=None, 
            rewards=[{0: 0.1, 1: 0.4, 2: 0.2, 3: 0.2, 4:0.1}, {1: 0.3, 2: 0.5, 3: 0.2}],
            episodic=False):
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)

        self.reward_distributions = tuple(map(lambda x: tuple(unzip(x.items())), rewards))
        self.episodic = episodic

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def _get_obs(self):
        return self._current_state

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_state = 0
        self.step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1
        match self._current_state:
            case 0:
                self._current_state = self.np_random.choice([1,2], p=self.probabilities[action])
                return self._current_state, 0, False, False, self._get_info()
            case s:
                reward = self.np_random.choice(self.reward_distributions[s-1][0], p=self.reward_distributions[s-1][1])
                self._current_state = 0
                return self._current_state, reward, self.episodic, False, self._get_info()    

