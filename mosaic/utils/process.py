
import os
import sys
import atexit
import psutil
import signal
import weakref
import threading
import multiprocessing

import mosaic


__all__ = ['subprocess']


_open_processes = weakref.WeakSet()


def subprocess(target):

    class Process:

        def __init__(self, *args, **kwargs):
            name = kwargs.pop('name', None)
            cpu_affinity = kwargs.pop('cpu_affinity', None)
            daemon = kwargs.pop('daemon', False)

            # _keep_child_alive is the write side of a pipe, which, when it is
            # closed, causes the read side of the pipe to unblock for reading. Note
            # that it is never closed directly. The write side is closed by the
            # kernel when our process exits, or possibly by the garbage collector
            # closing the file descriptor when the last reference to
            # _keep_child_alive goes away. We can take advantage of this fact to
            # monitor from the child and exit when the parent goes away unexpectedly
            # (for example due to SIGKILL). This variable is otherwise unused except
            # for the assignment here.
            parent_alive_pipe, self._keep_child_alive = multiprocessing.Pipe(duplex=False)

            # _parent_start_pipe will be used to signal that the child process is indeed alive
            # after we start it before we keep going forward.
            self._parent_start_pipe, child_start_pipe = multiprocessing.Pipe()

            self._parent_runtime = mosaic.runtime()
            self._mp_process = multiprocessing.Process(target=self._start_process,
                                                       name=name,
                                                       args=(target,
                                                             child_start_pipe,
                                                             parent_alive_pipe,
                                                             self._keep_child_alive,
                                                             cpu_affinity,
                                                             self._parent_runtime.uid,
                                                             self._parent_runtime.address,
                                                             self._parent_runtime.port,
                                                             args, kwargs))
            self._mp_process.daemon = daemon
            self._ps_process = None
            self._target = target
            self._obj = None

            self._state = 'pending'

        def __repr__(self):
            return "<Subprocess for %s, state=%s>" % (self._target, self._state)

        @property
        def state(self):
            return self._state

        def running(self):
            return self._state == 'running'

        def paused(self):
            return self._state == 'paused'

        def stopped(self):
            return self._state == 'stopped'

        def pause_process(self):
            if self._ps_process is not None:
                self._ps_process.suspend()
                self._state = 'paused'

        def start_process(self):
            if self._ps_process is not None:
                self._ps_process.resume()

            else:
                self._mp_process.start()
                self._ps_process = psutil.Process(self._mp_process.pid)
                self.cpu_load()

                _open_processes.add(self)

                self._parent_start_pipe.recv()
                self._parent_start_pipe.close()

            self._state = 'running'

        def _start_process(self, target,
                           child_start_pipe,
                           parent_alive_pipe, keep_child_alive,
                           cpu_affinity,
                           parent_id, parent_address, parent_port, args, kwargs):
            self._state = 'running'

            child_start_pipe.send(True)
            child_start_pipe.close()

            if sys.platform == 'linux' and cpu_affinity is not None:
                psutil.Process().cpu_affinity(cpu_affinity)

            keep_child_alive.close()
            self._immediate_exit_when_closed(parent_alive_pipe)

            mosaic.clear_runtime()

            self._target = target
            self._obj = self._target(*args, **kwargs,
                                     parent_id=parent_id,
                                     parent_address=parent_address,
                                     parent_port=parent_port)

            if hasattr(self._obj, 'run') and callable(self._obj.run):
                self._obj.run()

        @staticmethod
        def _immediate_exit_when_closed(parent_alive_pipe):
            def monitor_parent():
                try:
                    # The parent_alive_pipe should be held open as long as the
                    # parent is alive and wants us to stay alive. Nothing writes to
                    # it, so the read will block indefinitely.
                    parent_alive_pipe.recv()
                except EOFError:
                    # Parent process went away unexpectedly. Exit immediately. Could
                    # consider other exiting approaches here. My initial preference
                    # is to unconditionally and immediately exit. If we're in this
                    # state it is possible that a "clean" process exit won't work
                    # anyway - if, for example, the system is getting bogged down
                    # due to the running out of memory, exiting sooner rather than
                    # later might be needed to restore normal system function.
                    # If this is in appropriate for your use case, please file a
                    # bug.
                    os._exit(-1)

            thread = threading.Thread(target=monitor_parent)
            thread.daemon = True
            thread.start()

        def stop_process(self):
            if self._ps_process is not None:
                try:
                    self._ps_process.terminate()

                    if self in _open_processes:
                        _open_processes.remove(self)

                except (psutil.NoSuchProcess, OSError, RuntimeError):
                    pass

                self._state = 'stopped'

        def join_process(self, timeout=0.1):
            self._mp_process.join(timeout)

        def memory(self):
            if self._ps_process is not None:
                return self._ps_process.memory_info().rss

            return 0

        def cpu_load(self):
            if self._ps_process is not None:
                return self._ps_process.cpu_percent(interval=None)

            return 0

    return Process


def _close_processes():
    for process in list(_open_processes):
        process.stop_process()


def _close_processes_atsignal(signum, frame):
    _close_processes()

    os._exit(-1)


atexit.register(_close_processes)
signal.signal(signal.SIGINT, _close_processes_atsignal)
signal.signal(signal.SIGTERM, _close_processes_atsignal)