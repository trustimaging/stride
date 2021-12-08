
import os
import sys
import glob
import copy
import time
import pickle
import atexit
import inspect
import asyncio
import datetime
import functools
from collections import OrderedDict
from cached_property import cached_property

import mosaic

import _profile


__all__ = ['Profiler', 'GlobalProfiler', 'profiler',
           'global_profiler', 'skip_profile', 'no_profiler',
           'use_trace']


filter_modules = ['devito', 'sympy', 'matplotlib', 'traits', 'numpy', 'scipy', 'tkinter']
profiler = None
global_profiler = None


def _format_info(filename, lineno, func_name, cls_name=None):
    if cls_name:
        func_name = '%s.%s' % (cls_name, func_name)

    info = {
        'name': func_name,
        'filename': filename,
        'lineno': lineno,
    }

    return info


def _full_info(frame):
    func_code = frame.f_code
    func_name = func_code.co_name
    full_path = func_code.co_filename
    lineno = frame.f_lineno

    # first, let's whether we should even go on
    if full_path.startswith('<') and full_path.endswith('>'):
        return _format_info(full_path, lineno, func_name)

    path, filename = os.path.split(full_path)

    for module in filter_modules:
        if module in path:
            return _format_info(full_path, lineno, func_name)

    # then, let's see if it is a builtin
    if func_name.startswith('<') and func_name.endswith('>'):
        return _format_info(full_path, lineno, func_name)

    # otherwise, it could be directly defined in the module
    module = inspect.getmodule(frame)

    func = getattr(module, func_name, None)
    if func is not None and getattr(func, '__code__', None) is func_code:
        return _format_info(full_path, lineno, func_name)

    # if not, it might belong to a class at the first level of the
    # module
    for cls_name in dir(module):
        cls = getattr(module, cls_name)
        if not inspect.isclass(cls):
            continue

        # see if this class has a method with the name
        # we're looking for
        try:
            method = vars(cls)[func_name]
        except KeyError:
            continue

        # unwrap the method just in case there are any decorators
        try:
            method = inspect.unwrap(method)
        except ValueError:
            pass

        # see if this is the method that called us
        if getattr(method, '__code__', None) is func_code:
            return _format_info(full_path, lineno, func_name, cls_name)

    # if not at the top level, the class might be given as a free variable
    # within the locals
    if '__class__' in func_code.co_freevars:
        try:
            cls_name = frame.f_locals['__class__'].__name__
            return _format_info(full_path, lineno, func_name, cls_name)
        except KeyError:
            pass

    # if not a free variable, then it might be given as the first
    # argument to the function
    try:
        self_name = func_code.co_varnames[0]
    except IndexError:
        return _format_info(full_path, lineno, func_name)

    try:
        self_type = type(frame.f_locals[self_name])
    except KeyError:
        return _format_info(full_path, lineno, func_name)

    for cls in self_type.__mro__:
        # see if this class has a method with the name
        # we're looking for
        try:
            method = vars(cls)[func_name]
        except KeyError:
            continue

        # assume that this is the method that called
        cls_name = cls.__name__
        return _format_info(full_path, lineno, func_name, cls_name)

    return _format_info(full_path, lineno, func_name)


def trace(frame, event, arg):
    frame.f_trace_lines = False

    if not len(profiler.active_traces):
        return trace

    if event not in ['call', 'return', 'exception']:
        return

    if event == 'exception':
        return trace

    if event == 'call':
        frame_id = profiler.f_call(frame)

        if frame_id:
            return trace

    elif event == 'return':
        profiler.f_return(frame)


def profiled_task_factory(loop, coro):
    if profiler.tracing:
        sys.settrace(None)

    # create task
    child_task = asyncio.tasks.Task(coro, loop=loop)

    if not profiler.tracing:
        return child_task

    if not len(profiler.active_traces):
        profiler.maybe_trace()
        return child_task

    # find the outer and inner frames,
    # delete the node corresponding to the current frame
    trace_id, outer_frame_id, curr_frame_id = profiler._del_node()

    if outer_frame_id is None:
        profiler.maybe_trace()
        return child_task

    if not hasattr(coro, 'cr_frame') and not hasattr(coro, 'gi_frame'):
        profiler.maybe_trace()
        return child_task

    try:
        frame = coro.cr_frame
    except AttributeError:
        frame = coro.gi_frame

    # substitute it with a new one for the coro
    profiler._new_node(trace_id, outer_frame_id, frame, timeit=False)

    profiler.maybe_trace()

    return child_task


def dict_diff(dict_1, dict_2):
    diff_dict = OrderedDict()

    for key, value in dict_2.items():
        if key not in dict_1:
            diff_dict[key] = value

        elif isinstance(value, dict):
            diff_dict[key] = dict_diff(dict_1[key], value)

    return diff_dict


def dict_update(dict_1, dict_2):
    for key, value in dict_2.items():
        if isinstance(value, dict):
            dict_1[key] = dict_update(dict_1.get(key, {}), value)

        else:
            dict_1[key] = value

    return dict_1


class Profiler:
    """
    The profiler controls profiling for the local runtime, including starting and stopping
    the profile, and starting new traces.

    """

    def __init__(self):
        self.nodes = OrderedDict()
        self.active_nodes = dict()
        self.async_nodes = set()
        self.hash_nodes = dict()
        self.skip_nodes = dict()
        self.task_nodes = dict()
        self.traces = dict()
        self.active_traces = dict()
        self.t_start = None
        self.t_end = None
        self.t_elapsed = None
        self._tracing = False
        self._runtime_id = None

    def maybe_trace(self):
        if self.tracing and len(self.active_traces):
            sys.settrace(trace)

    def maybe_stop_trace(self):
        if not len(self.active_traces):
            sys.settrace(None)

    def clear(self):
        """
        Clear profiler.

        Returns
        -------

        """
        self.nodes = OrderedDict()
        self.active_nodes = dict()
        self.async_nodes = set()
        self.hash_nodes = dict()
        self.skip_nodes = dict()
        self.traces = dict()
        self.active_traces = dict()
        self.t_start = None
        self.t_end = None
        self.t_elapsed = None

    def start(self):
        """
        Start profiling.

        Returns
        -------

        """
        self.clear()
        self.t_start = time.time()
        self._tracing = True

        runtime = mosaic.runtime()
        self._runtime_id = runtime.uid if runtime is not None else 'head'

        loop = asyncio.get_event_loop()
        loop.set_task_factory(profiled_task_factory)

        if global_profiler is not None:
            global_profiler.start()

    def stop(self):
        """
        Stop profiling.

        Returns
        -------

        """
        for trace_id in self.active_traces.keys():
            self.stop_trace(trace_id)

        self.t_end = time.time()
        self.t_elapsed = self.t_end - self.t_start
        self._tracing = False

        sys.settrace(None)
        
        if global_profiler is not None:
            global_profiler.stop()

    def start_trace(self, trace_id=None, level=1):
        """
        Start a new profiling trace.

        Parameters
        ----------
        trace_id : str, optional
            Optional trace_id for this trace.
        level : int, optional
            Level at which the trace should start.

        Returns
        -------

        """
        if not self.tracing:
            return

        root_frame = sys._getframe(level)

        if trace_id is None:
            trace_id = self._frame_id(root_frame)
        else:
            self.hash_nodes[trace_id] = 1

        t_start = time.time()
        new_node = OrderedDict(name='trace:%d' % len(self.traces),
                               frame_id=trace_id,
                               trace_id=trace_id,
                               t_start=t_start)

        try:
            outer_frame_id = root_frame.f_locals['__frame_id__']
            outer_trace_id = root_frame.f_locals['__trace_id__']

            root_frame.f_locals['__prev_frame_id__'] = outer_frame_id
            root_frame.f_locals['__prev_trace_id__'] = outer_trace_id

            self.active_nodes[outer_frame_id][trace_id] = new_node
        except KeyError:
            self.nodes[trace_id] = new_node

        root_frame.f_locals['__frame_id__'] = trace_id
        root_frame.f_locals['__trace_id__'] = trace_id
        root_frame.f_trace_lines = False

        self.active_nodes[trace_id] = new_node
        self.traces[trace_id] = new_node
        self.active_traces[trace_id] = new_node

        is_async = _profile.is_async(root_frame)
        if is_async:
            self.async_nodes.add(trace_id)

        self.maybe_trace()

        return trace_id

    def stop_trace(self, trace_id=None, level=1):
        """
        Stop a profiling trace.

        Returns
        -------

        """
        _trace_id, _, _ = self._del_node(level=level)

        if not self.tracing:
            return

        root_frame = sys._getframe(level)

        try:
            outer_frame_id = root_frame.f_locals['__prev_frame_id__']
            outer_trace_id = root_frame.f_locals['__prev_trace_id__']

            root_frame.f_locals['__frame_id__'] = outer_frame_id
            root_frame.f_locals['__trace_id__'] = outer_trace_id
        except KeyError:
            pass

        trace_id = trace_id or _trace_id

        try:
            node = self.active_nodes[trace_id]
            del self.active_nodes[trace_id]
            del self.active_traces[trace_id]
        except KeyError:
            return

        t_end = time.time()
        node['t_end'] = t_end
        node['t_elapsed'] = node['t_end'] - node['t_start']

        if trace_id in self.async_nodes:
            self.async_nodes.remove(trace_id)

        self.maybe_stop_trace()

    def frame_info(self, level=1):
        frame = sys._getframe(level)

        try:
            trace_id = frame.f_locals['__trace_id__']
            frame_id = frame.f_locals['__frame_id__']
        except KeyError:
            return None, None

        return trace_id, frame_id

    @property
    def tracing(self):
        """
        Whether or not the profiler is activated.

        """
        return self._tracing

    def f_call(self, frame):
        """
        Event fired when function is called.

        Parameters
        ----------
        frame

        Returns
        -------

        """
        if not self.tracing:
            return

        try:
            outer_frame_id = frame.f_back.f_locals['__frame_id__']
            trace_id = frame.f_back.f_locals['__trace_id__']
        except KeyError:
            return

        return self._new_node(trace_id, outer_frame_id, frame)

    def f_return(self, frame):
        """
        Event fired when function returns.

        Parameters
        ----------
        frame

        Returns
        -------

        """
        if not self.tracing:
            return

        try:
            frame_id = frame.f_locals['__frame_id__']
            trace_id = frame.f_locals['__trace_id__']
        except KeyError:
            return

        is_async = _profile.is_async(frame)
        is_suspended = is_async and _profile.is_suspended(frame)

        if is_suspended:
            return

        if is_async and frame_id in self.async_nodes:
            self.async_nodes.remove(frame_id)

        try:
            node = self.active_nodes[frame_id]
            del self.active_nodes[frame_id]
        except KeyError:
            return

        t_end = time.time()
        node['t_end'] = t_end
        node['t_elapsed'] = node['t_end'] - node['t_start']

    def _new_node(self, trace_id, outer_frame_id, frame, timeit=True):
        if outer_frame_id not in self.active_nodes:
            return

        # assign or retrieve a frame ID
        try:
            frame_id = frame.f_locals['__frame_id__']
        except KeyError:
            frame_id = self._frame_id(frame)
            frame.f_locals['__frame_id__'] = frame_id

        # assign a trace ID
        frame.f_locals['__trace_id__'] = trace_id

        # check for suspended async frame
        is_async = _profile.is_async(frame)
        if is_async:
            if frame_id in self.async_nodes:
                new_node = self.active_nodes[frame_id]

                if new_node.get('t_start') is None:
                    new_node['t_start'] = time.time()

                return frame_id

            self.async_nodes.add(frame_id)

        # create new node
        frame_info = _full_info(frame)
        t_start = time.time() if timeit else None
        new_node = OrderedDict(**frame_info,
                               frame_id=frame_id,
                               trace_id=trace_id,
                               t_start=t_start)

        try:
            self.active_nodes[outer_frame_id][frame_id] = new_node
            self.active_nodes[frame_id] = new_node
        except KeyError:
            pass

        return frame_id

    def _del_node(self, level=1):
        curr_frame = sys._getframe(level)
        outer_frame = sys._getframe(level+1)

        try:
            outer_frame_id = outer_frame.f_locals['__frame_id__']
            trace_id = outer_frame.f_locals['__trace_id__']
        except KeyError:
            return None, None, None

        try:
            curr_frame_id = curr_frame.f_locals['__frame_id__']

            del self.active_nodes[outer_frame_id][curr_frame_id]
            del self.active_nodes[curr_frame_id]
        except KeyError:
            return trace_id, outer_frame_id, None

        return trace_id, outer_frame_id, curr_frame_id

    def _frame_id(self, frame):
        frame_id = str(id(frame))

        if frame_id not in self.hash_nodes:
            self.hash_nodes[frame_id] = 0

        count = self.hash_nodes[frame_id] = self.hash_nodes[frame_id] + 1

        return '%s.%s.%d' % (self._runtime_id, frame_id, count)


profiler = Profiler()


def skip_profile(*args, **kwargs):
    """
    Skip profiling for a decorated function:

    >>> @skip_profile
    >>> def function_not_profiled():
    >>>     pass

    Parameters
    ----------
    stop_trace : bool, optional
        Whether to stop the system trace, defaults to False.

    Returns
    -------

    """

    def make_wrapper(func, stop_trace):
        profiler._del_node()

        @functools.wraps(func)
        def wrapper(*_args, **_kwargs):
            with no_profiler('skip_profile:%s' % func.__name__,
                             level=2,
                             stop_trace=stop_trace):
                ret = func(*_args, **_kwargs)

            return ret

        return wrapper

    if len(args) == 1 and callable(args[0]):
        profiler._del_node()
        return make_wrapper(args[0], False)

    else:
        def _skip_profile(func):
            profiler._del_node()
            return make_wrapper(func, kwargs.get('stop_trace', False))

        return _skip_profile


class no_profiler:
    """
    Skip profiling for a code block:

    >>> with no_profiler():
    >>>     pass

    Parameters
    ----------
    stop_trace : bool, optional
        Whether to stop the system trace, defaults to False.

    Returns
    -------

    """
    def __init__(self, name='no_profiler', level=1, stop_trace=False):
        profiler._del_node(level=level)

        self._name = name
        self._level = level
        self._stop_trace = stop_trace
        self._node = None
        self._outer_node = None
        self._frame_id = None

    def __enter__(self):
        trace_id, outer_frame_id, curr_frame_id = profiler._del_node(level=self._level)

        if outer_frame_id:
            if self._stop_trace:
                sys.settrace(None)

            t_start = time.time()
            new_node = OrderedDict(name=self._name, t_start=t_start)

            self._node = new_node
            profiler.active_nodes[outer_frame_id][curr_frame_id] = new_node

            self._frame_id = outer_frame_id
            self._outer_node = profiler.active_nodes[outer_frame_id]
            del profiler.active_nodes[outer_frame_id]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._frame_id is not None:
            if self._stop_trace:
                profiler.maybe_trace()

            t_end = time.time()
            self._node['t_end'] = t_end
            self._node['t_elapsed'] = self._node['t_end'] - self._node['t_start']

            profiler.active_nodes[self._frame_id] = self._outer_node


class use_trace:
    def __init__(self, trace_id, outer_frame_id=None, level=1):
        profiler._del_node(level=level)

        self._trace_id = trace_id
        self._outer_frame_id = outer_frame_id
        self._level = level

        self._prev_trace_id = None
        self._prev__outer_frame_id = None
        self._frame = None

    def __enter__(self):
        if not profiler.tracing:
            return

        trace_id, outer_frame_id, curr_frame_id = profiler._del_node(level=self._level)

        self._frame = sys._getframe(self._level)

        self._frame.f_locals['__trace_id__'] = self._trace_id
        if self._outer_frame_id:
            self._frame.f_locals['__frame_id__'] = self._outer_frame_id

        self._prev_trace_id = trace_id
        self._prev__outer_frame_id = outer_frame_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not profiler.tracing:
            return

        profiler._del_node(level=self._level)

        self._frame.f_locals['__trace_id__'] = self._prev_trace_id
        if self._outer_frame_id:
            self._frame.f_locals['__frame_id__'] = self._prev__outer_frame_id


class LocalProfiler:

    def __init__(self):
        self.profiles = OrderedDict()
        self._last_profile = OrderedDict()
        self._last_part = -1
        self.t_start = None
        self.t_end = None
        self.t_elapsed = None

        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.filename = '%s.profile' % now

    def clear(self):
        self.profiles = OrderedDict()
        self._last_profile = OrderedDict()
        self._last_part = -1
        self.t_start = None
        self.t_end = None
        self.t_elapsed = None

    def start(self):
        self.clear()

        runtime = mosaic.runtime()
        self.profiles[runtime.uid] = profiler.nodes

        self.t_start = time.time()

        # loop = mosaic.get_event_loop()
        # loop.interval(self.append, filename=self.filename, interval=30)

    def stop(self):
        self.t_end = time.time()
        self.t_elapsed = self.t_end - self.t_start

    @skip_profile(stop_trace=True)
    def update(self, sender_id, profiler_update):
        if sender_id not in self.profiles:
            self.profiles[sender_id] = OrderedDict()

        self.profiles[sender_id] = dict_update(self.profiles[sender_id], profiler_update)

    @skip_profile(stop_trace=True)
    def dump(self, *args, **kwargs):
        description = dict()
        description['profiles'] = self.profiles
        description['t_start'] = self.t_start
        description['t_end'] = self.t_end
        description['t_elapsed'] = self.t_elapsed

        filename = kwargs.pop('filename')

        with open(filename, 'wb') as file:
            pickle.dump(description, file, protocol=pickle.HIGHEST_PROTOCOL)

        self._last_profile = copy.deepcopy(self.profiles)

    @skip_profile(stop_trace=True)
    def load(self, *args, **kwargs):
        filename = kwargs.pop('filename')

        if filename.endswith('.parts'):
            path = filename
            filename = filename.split('/')[-1][:-6]

            parts = glob.glob(os.path.join(path, '%s.*' % filename))
            parts.sort()

            for part in parts:
                with open(part, 'rb') as file:
                    update = pickle.load(file)

                self.t_start = update.get('t_start')
                self.t_end = update.get('t_end')
                self.t_elapsed = update.get('t_elapsed')

                self.profiles = dict_update(self.profiles, update['profiles'])

        else:
            with open(filename, 'rb') as file:
                update = pickle.load(file)

            self.t_start = update.get('t_start')
            self.t_end = update.get('t_end')
            self.t_elapsed = update.get('t_elapsed')

            self.profiles = update['profiles']

    @skip_profile(stop_trace=True)
    def append(self, *args, **kwargs):
        path = '%s.parts' % self.filename
        if not os.path.exists(path):
            os.makedirs(path)

        self._last_part += 1
        filename = os.path.join(path, '%s.%d' % (self.filename, self._last_part))

        description = dict()
        description['profiles'] = self.get_update()
        description['t_start'] = self.t_start
        description['t_end'] = self.t_end
        description['t_elapsed'] = self.t_elapsed

        with open(filename, 'wb') as file:
            pickle.dump(description, file, protocol=pickle.HIGHEST_PROTOCOL)

    @skip_profile(stop_trace=True)
    def get_update(self):
        current_profile = self.profiles
        profile_update = dict_diff(self._last_profile, current_profile)

        for runtime_id, runtime in self.profiles.items():
            for trace_id, trace_node in runtime.items():
                if trace_id not in profile_update[runtime_id]:
                    profile_update[runtime_id][trace_id] = OrderedDict()

                profile_update[runtime_id][trace_id]['t_end'] = trace_node.get('t_end')
                profile_update[runtime_id][trace_id]['t_elapsed'] = trace_node.get('t_elapsed')

        self._last_profile = current_profile

        return profile_update


class RemoteProfiler:

    def __init__(self, runtime_id='monitor'):
        self._runtime_id = runtime_id
        self._last_profile = OrderedDict()

    @cached_property
    def remote_runtime(self):
        runtime = mosaic.runtime()
        return runtime.proxy(self._runtime_id)

    def clear(self):
        self._last_profile = OrderedDict()

    def start(self):
        self.clear()

    def stop(self):
        pass

    @skip_profile(stop_trace=True)
    def update(self):
        # start = time.time()
        profiler_update = self.get_update()
        # print(mosaic.runtime().uid, 1, time.time()-start)

        if not len(profiler_update.keys()):
            return

        self.remote_runtime.recv_profile(profiler_update=profiler_update, as_async=False)

        return profiler_update

    @skip_profile(stop_trace=True)
    def get_update(self):
        current_profile = profiler.nodes
        profiler_update = dict_diff(self._last_profile, current_profile)

        for trace_id, trace_node in profiler.nodes.items():
            if trace_id not in profiler_update:
                profiler_update[trace_id] = OrderedDict()

            profiler_update[trace_id]['t_end'] = trace_node.get('t_end')
            profiler_update[trace_id]['t_elapsed'] = trace_node.get('t_elapsed')

        self._last_profile = current_profile

        return profiler_update


class GlobalProfiler:
    """
    The global profiler keeps the different endpoints of the runtime in contact,
    so that local profiles can be consolidated into a single, global one.

    """

    def __init__(self):
        self.profiler = None
        self.mode = None

    def start(self):
        self.profiler.start()
        
    def stop(self):
        self.profiler.stop()

    def set_local(self):
        self.profiler = LocalProfiler()
        self.mode = 'local'

    def set_remote(self, runtime_id='monitor'):
        self.profiler = RemoteProfiler(runtime_id=runtime_id)
        self.mode = 'remote'

    def send_profile(self):
        if self.mode != 'remote':
            return

        self.profiler.update()

    def get_profile(self):
        if self.mode != 'remote':
            return

        return self.profiler.get_update()

    def recv_profile(self, sender_id, profiler_update):
        if self.mode != 'local':
            return

        self.profiler.update(sender_id, profiler_update)

    def dump(self, *args, **kwargs):
        if self.mode != 'local':
            return

        self.profiler.dump(*args, **kwargs)

    def load(self, *args, **kwargs):
        if self.mode != 'local':
            return

        self.profiler.load(*args, **kwargs)

    def append(self, *args, **kwargs):
        if self.mode != 'local':
            return

        self.profiler.append(*args, **kwargs)


global_profiler = GlobalProfiler()


def _stop_tracing():
    sys.settrace(None)


atexit.register(_stop_tracing)
