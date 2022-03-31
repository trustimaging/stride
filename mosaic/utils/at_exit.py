
import os
import signal
import atexit


__all__ = ['at_exit']


class AtExit:

    def __init__(self):
        self.funcs = set()

    def add(self, func):
        self.funcs.add(func)

    def remove(self, func):
        self.funcs.remove(func)


at_exit = AtExit()


def _close():
    for func in list(at_exit.funcs):
        try:
            func()
        except Exception:
            pass


def _close_atsignal(signum, frame):
    _close()
    os._exit(-1)


atexit.register(_close)
signal.signal(signal.SIGINT, _close_atsignal)
signal.signal(signal.SIGTERM, _close_atsignal)
