
from abc import ABC, abstractmethod


__all__ = ['LocalOptimiser']


class LocalOptimiser(ABC):

    def __init__(self, variable, **kwargs):
        self.variable = variable

    @abstractmethod
    def apply(self, grad, **kwargs):
        pass
