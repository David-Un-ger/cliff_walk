from abc import ABC, abstractmethod
from gymnasium import Env


class Strategy(ABC):
    '''
    Abstract strategy class.
    '''
    @abstractmethod
    def step(self, env: Env, observation: int, train: bool = False):
        pass
