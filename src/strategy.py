from typing import Tuple
from abc import ABC, abstractmethod
from matplotlib.pyplot import Axes

from gymnasium import Env


class Strategy(ABC):
    '''
    Abstract strategy class.
    '''

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(self, env: Env, max_nr_steps: int = 10000):
        pass

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

    @abstractmethod
    def plot_training(self, ax: Axes):
        pass

    @abstractmethod
    def step(self,
             env: Env,
             observation: int,
             train: bool = False) -> Tuple[Env, int]:
        pass
