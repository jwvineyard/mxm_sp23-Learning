import numpy as np
import pandas as pd
from functools import reduce
from itertools import product



class DynaQ:
    def __init__(self, action_space, learning_rate=0.01, discount_rate=0.95, exploration_rate=0.1):
        self.actions = action_space
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.model = pd.DataFrame(columns=self.actions, dtype=np.float64)
    
    