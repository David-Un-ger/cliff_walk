import numpy as np
from typing import Tuple

from strategy import Strategy
from gymnasium import Env


class QTableStrategy(Strategy):
    '''
    Implements the one step Q-table strategy.
    '''

    def __init__(self, nr_observations: int, nr_actions):
        self.q_table = np.zeros((nr_observations, nr_actions))

    def step(self, env: Env, curr_observation: int, train: bool = False) -> Tuple[Env, int]:
        '''
        The action with the highest rating in the Q table for the given
        observation is chosen and executed in the given environment.

        If training is activated the executed action is rated in the Q table
        based its reward and the rating of the highest rated action it the next state.
        '''
        action = np.argmax(self.q_table[curr_observation])
        new_observation, reward, terminated, _, _ = env.step(action)
        if terminated:
            reward = 100

        if train:
            self.q_table[curr_observation, action] = reward + \
                np.max(self.q_table[new_observation])

        if terminated or reward == -100:
            new_observation, _ = env.reset()

        return env, new_observation

    def __str__(self):
        return f"{self.q_table}"
