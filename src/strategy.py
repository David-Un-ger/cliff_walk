from typing import Tuple
from abc import ABC, abstractmethod

from gymnasium import Env


class Strategy(ABC):
    '''
    Abstract strategy class.
    '''

    def __init__(self):
        self.training_steps_taken = 0

    @abstractmethod
    def step(self,
             env: Env,
             observation: int,
             train: bool = False) -> Tuple[Env, int]:
        pass
