from typing import Tuple
from abc import ABC, abstractmethod
from matplotlib.pyplot import Axes
import pandas as pd

from gymnasium import Env


class Strategy(ABC):
    '''
    Abstract strategy class.
    '''

    def __init__(self, name: str):
        self.name = name

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

    def run(self, env: Env, max_nr_steps: int = 1000):
        observation, _ = env.reset()
        for step_idx in range(max_nr_steps):
            env, observation, cancelled, terminated = self.step(env,
                                                                observation,
                                                                train=False)

            if cancelled or terminated:
                print(
                    f"{self.name}: {'Reached the goal' if terminated else 'Fell of the cliff'} in {step_idx+1}"
                )
                break

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

    @abstractmethod
    def step(self,
             env: Env,
             observation: int,
             train: bool = False) -> Tuple[Env, int]:
        pass
