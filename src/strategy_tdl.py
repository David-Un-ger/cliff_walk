from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from strategy import Strategy
from gymnasium import Env


class TDLStrategy(Strategy, ABC):
    '''
    Abstract base class for temporal difference learning strategies. Derivates need to 
    implement:
        - update_q_table(self, action: int, curr_obs: int, new_observation: int,
                       reward: int):
    '''

    def __init__(self,
                 nr_observations: int,
                 nr_actions: int,
                 name: str = "Temporal Difference Learning"):
        super().__init__(name)
        self.q_table = np.zeros((nr_observations, nr_actions))

    def get_action(self, curr_obs: int) -> int:
        '''
        Gets the highest rated action for the current observation from the
        Q-table.
        '''
        return np.argmax(self.q_table[curr_obs])

    @abstractmethod
    def update_q_table(self, action: int, curr_obs: int, new_observation: int,
                       reward: int):
        pass

    def step(self,
             env: Env,
             curr_obs: int,
             train: bool = False) -> Tuple[Env, int]:
        '''
        The action with the highest rating in the Q table for the given
        observation is chosen and executed in the given environment.

        If training is activated the Q-table is updated based on the reward.
        '''
        action = self.get_action(curr_obs)

        new_observation, reward, terminated, _, _ = env.step(action)

        cancelled = reward == -100

        if terminated:
            reward = 100

        if train:
            self.update_q_table(action, curr_obs, new_observation, reward)

        return env, new_observation, cancelled, terminated

    def __str__(self):
        return f"{self.q_table}"


class SimpleQLearningStrategy(TDLStrategy):
    '''
    Implements a simplified Q-learining strategy by using a simplified update
    function for the Q-table.
    '''

    def __init__(self, nr_observations: int, nr_actions):
        super().__init__(nr_observations, nr_actions, "Simple Q Learning")

    def update_q_table(self, action: int, curr_obs: int, new_observation: int,
                       reward: int):
        self.q_table[curr_obs,
                     action] = reward + np.max(self.q_table[new_observation])


class QLearningStrategy(TDLStrategy):
    '''
    Implements the one step Q-table strategy.
    '''

    def __init__(self, nr_observations: int, nr_actions, alpha: int,
                 gamma: int):
        super().__init__(nr_observations, nr_actions, "Q Learning")
        self.alpha = alpha
        self.gamma = gamma

    def update_q_table(self, action: int, curr_obs: int, new_observation: int,
                       reward: int):
        self.q_table[
            curr_obs, action] = self.q_table[curr_obs, action] + self.alpha * (
                reward + self.gamma * np.max(self.q_table[new_observation]) -
                self.q_table[curr_obs, action])
