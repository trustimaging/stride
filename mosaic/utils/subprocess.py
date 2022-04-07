
import os
import sys
import psutil
import weakref
import functools
import threading
import multiprocessing

import mosaic
from .at_exit import at_exit

try:
    import daemon
except (ModuleNotFoundError, ImportError):
    DAEMON_AVAILABLE = False
else:
    DAEMON_AVAILABLE = True


__all__ = ['subprocess']


_open_processes = weakref.WeakSet()


class Subprocess:
    """
    Class to manage a subprocess that executes a target function.

    It manages the creation and destruction of the process, and can be used
    to collect statistics about it.

    Parameters
    ----------
    name : str, optional
        Name to give to the subprocess.
    target : callable
        Target function to be executed in the subprocess.
    cpu_affinity : list, optional
        List of CPUs to set the affinity of the process, defaults to None.
    daemon : bool, optional
        Whether to start the subprocess as a daemon, defaults to False.

    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        target = kwargs.pop('target', None)
        cpu_affinity = kwargs.pop('cpu_affinity', None)
        is_daemon = kwargs.pop('daemon', False)

        if target is None or not callable(target):
            raise ValueError('A subprocess needs to be provided a target function.')

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
        if self._parent_runtime is not None:
            parent_args = (self._parent_runtime.uid,
                           self._parent_runtime.address,
                           self._parent_runtime.port,)

        else:
            parent_args = (None, None, None)

        self._mp_process = multiprocessing.Process(target=self._start_process,
                                                   name=name,
                                                   args=(target,
                                                         is_daemon,
                                                         child_start_pipe,
                                                         parent_alive_pipe,
                                                         self._keep_child_alive,
                                                         cpu_affinity,
                                                         *parent_args,
                                                         args, kwargs))
        self._ps_process = None
        self._target = target
        self._obj = None

        self._state = 'pending'

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return "<Subprocess for %s, state=%s>" % (self._target, self._state)

    @property
    def pid(self):
        """
        Process PID.

        """
        return self._mp_process.pid

    @property
    def state(self):
        """
        Current state of the process.

        It could be ``pending``, ``running``, ``paused`` or ``stopped``.

        """
        return self._state

    def running(self):
        """
        Whether or not the process is running.

        Returns
        -------
        bool

        """
        return self._state == 'running'

    def paused(self):
        """
        Whether or not the process is paused.

        Returns
        -------
        bool

        """
        return self._state == 'paused'

    def stopped(self):
        """
        Whether or not the process is stopped.

        Returns
        -------
        bool

        """
        return self._state == 'stopped'

    def pause_process(self):
        """
        Pause the subprocess.

        Returns
        -------

        """
        if self._ps_process is not None:
            self._ps_process.suspend()
            self._state = 'paused'

    def start_process(self):
        """
        Start or resume the subprocess.

        Returns
        -------

        """
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
                       is_daemon,
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
        if not is_daemon:
            self._immediate_exit_when_closed(parent_alive_pipe)

        mosaic.clear_runtime()

        try:
            if is_daemon:
                if not DAEMON_AVAILABLE:
                    raise RuntimeError('Tried to create a daemon subprocess with '
                                       'no "python-daemon" available')

                from .logger import _stdout, _stderr
                daemon_context = daemon.DaemonContext(detach_process=True,
                                                      stdout=_stdout,
                                                      stderr=_stderr)
                daemon_context.open()

            self._target = target
            self._obj = self._target(*args, **kwargs,
                                     parent_id=parent_id,
                                     parent_address=parent_address,
                                     parent_port=parent_port)

            if hasattr(self._obj, 'run') and callable(self._obj.run):
                self._obj.run()

        finally:
            if is_daemon:
                daemon_context.close()

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
        """
        Stop the subprocess.

        Returns
        -------

        """
        if self._ps_process is not None:
            try:
                self._ps_process.terminate()

                if self in _open_processes:
                    _open_processes.remove(self)

            except (psutil.NoSuchProcess, OSError, RuntimeError):
                pass

            self._state = 'stopped'

    def join_process(self, timeout=0.1):
        """
        Join the subprocess.

        Parameters
        ----------
        timeout : float, optional
            Time to wait to join, defaults to 0.1.

        Returns
        -------

        """
        self._mp_process.join(timeout)

    def memory(self):
        """
        Amount of RSS memory being consumed by the process.

        Returns
        -------
        float
            RSS memory.

        """
        if self._ps_process is not None:
            # OSX does not allow accessing information on external processes
            try:
                return self._ps_process.memory_info().rss
            except psutil.AccessDenied:
                pass

        return 0

    def cpu_load(self):
        """
        CPU load as a percentage.

        Returns
        -------
        float
            CPU load.

        """
        if self._ps_process is not None:
            # OSX does not allow accessing information on external processes
            try:
                return self._ps_process.cpu_percent(interval=None)
            except psutil.AccessDenied:
                pass

        return 0

    def cpu_affinity(self, cpus):
        """
        Set CPU affinity for this process.

        Parameters
        ----------
        cpus : list
            List of pinned CPUs.

        Returns
        -------

        """
        try:
            import numa
            numa_available = numa.info.numa_available()
        except Exception:
            numa_available = False

        if numa_available:
            numa.schedule.run_on_cpus(self.pid, *cpus)
        else:
            self._ps_process.cpu_affinity(cpus)


def subprocess(target):
    """
    A decorator that will execute a target function in a subprocess. The generated subprocess
    will be encapsulated in a class that has methods to manage the subprocess.

    Parameters
    ----------
    target : callable
        Target function to be executed in the subprocess

    Returns
    -------
    Subprocess
        Instance of class Subprocess.

    """

    return functools.partial(Subprocess, target=target)


def _close_processes():
    for process in list(_open_processes):
        process.stop_process()


at_exit.add(_close_processes)
