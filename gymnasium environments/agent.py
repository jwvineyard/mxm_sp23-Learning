from tqdm import tqdm
import numpy as np

class Agent:
    def get_action(self, observation): pass

    def update(self,
        observation,
        action,
        reward,
        terminated,
        truncated,
        next_observation): pass
    
def train_episode(agent, environment, seed=None):
    obs, info = environment.reset(seed=seed)
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = environment.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, truncated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

def train(agent, environment, n_episodes=2000):
    for episode in tqdm(range(n_episodes)):
        train_episode(agent, environment, seed=np.random.randint(0, 100000000))

def rate_decay(base=0.95, decay=0.999995):
    return lambda t: base * (decay ** t)
