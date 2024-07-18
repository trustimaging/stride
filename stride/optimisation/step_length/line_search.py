
from abc import abstractmethod


__all__ = ['LineSearch']


class LineSearch:

    @abstractmethod
    def init_search(self, variable, direction, **kwargs):
        pass

    @abstractmethod
    def next_step(self, variable, direction, **kwargs):
        pass

    def forward_test(self, variable, direction, **kwargs):
        pass
