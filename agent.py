class Agent:
    def get_action(self, observation): pass

    def update(self,
        observation,
        action,
        reward,
        terminated,
        truncated,
        next_observation): pass
    
def train_episode(agent, environment):
    obs, info = environment.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = environment.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, truncated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

