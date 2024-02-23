
from abc import abstractmethod


__all__ = ['LineSearch']


class LineSearch:

    @abstractmethod
    async def init_search(self, variable, direction, **kwargs):
        pass

    @abstractmethod
    async def next_step(self, variable, direction, **kwargs):
        pass

    async def forward_test(self, variable, direction, **kwargs):
        pass
