from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from matplotlib.pyplot import Axes
import pandas as pd

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

    def train(self, env: Env, max_nr_steps: int = 10000):
        observation, _ = env.reset()
        steps_since_reset = 0
        self.train_sequences = list()
        for train_step_idx in range(max_nr_steps):
            env, observation, cancelled, terminated = self.step(env,
                                                                observation,
                                                                train=True)
            steps_since_reset += 1

            if cancelled or terminated:
                print(
                    f"{self.name}: {'Reached the goal' if terminated else 'Fell of the cliff'} in {steps_since_reset} steps after {train_step_idx+1} training steps"
                )
                self.train_sequences.append({
                    "steps_since_reset": steps_since_reset,
                    "reached_goal": terminated,
                    "training_step": train_step_idx + 1
                })
                steps_since_reset = 0
                observation, _ = env.reset()

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

    def plot_training(self, ax: Axes):
        df_training_result = pd.DataFrame(self.train_sequences)
        df_training_result.plot(x="training_step",
                                y="steps_since_reset",
                                ax=ax,
                                linestyle="--",
                                color="grey",
                                zorder=1,
                                label=None)

        df_goal = df_training_result[df_training_result.reached_goal == True]
        df_cliff = df_training_result[df_training_result.reached_goal == False]

        df_goal.plot.scatter(x="training_step",
                             y="steps_since_reset",
                             s=30,
                             ax=ax,
                             marker="o",
                             color="green",
                             zorder=2,
                             label="Reached the goal")

        df_cliff.plot.scatter(x="training_step",
                              y="steps_since_reset",
                              s=30,
                              ax=ax,
                              marker="x",
                              color="red",
                              zorder=2,
                              label="Fell off the cliff")

        ax.grid("on")
        ax.set_axisbelow(True)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Steps since last reset")
        ax.set_title(
            f"{self.name} (min. steps: {df_goal.steps_since_reset.min()})")

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
