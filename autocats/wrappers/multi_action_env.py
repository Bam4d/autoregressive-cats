import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

class MultiActionEnv(gym.Wrapper):

    def __init__(self, env, actions_per_step):
        super().__init__(env)

        # multiply the action space by the number of actions
        if isinstance(self.action_space, Discrete):
            action_space = MultiDiscrete(np.tile(self.action_space.n, actions_per_step))
        else:
            action_space = MultiDiscrete(np.tile(self.action_space.nvec, actions_per_step))

        # if self.player_count > 1:
        #     self.action_space = MultiAgentActionSpace([action_space for _ in range(self.player_count)])

        self.action_space = action_space
