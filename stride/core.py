
import uuid
import asyncio
import inspect
import numpy as np
from abc import abstractmethod
from collections import OrderedDict

import mosaic
from mosaic import types
from mosaic.core.base import CMDBase
from mosaic.core import TaskProxy


__all__ = ['Variable', 'Operator']


async def _maybe_sum(a, b):
    if isinstance(a, types.awaitable_types):
        a = await a.result()

    if isinstance(b, types.awaitable_types):
        b = await b.result()

    if b is None:
        return a

    elif a is None:
        return b

    else:
        if isinstance(a, tuple) and not isinstance(b, tuple):
            return a[0] + b, a[1]
        elif not isinstance(a, tuple) and isinstance(b, tuple):
            return b[0] + a, b[1]
        elif isinstance(a, tuple) and isinstance(b, tuple):
            return a[0] + b[0], a[1] + b[1]

        return a + b


class no_grad:

    def __init__(self, *args, **kwargs):
        self.arg_flags = dict()
        self.kwarg_flags = dict()

        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        for index, variable in zip(range(len(self._args)), self._args):
            if hasattr(variable, 'needs_grad'):
                self.arg_flags[index] = variable.needs_grad
                variable.needs_grad = False

        for key, variable in self._kwargs.items():
            if hasattr(variable, 'needs_grad'):
                self.kwarg_flags[key] = variable.needs_grad
                variable.needs_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for index, variable in zip(range(len(self._args)), self._args):
            if hasattr(variable, 'needs_grad'):
                variable.needs_grad = self.arg_flags[index]

        for key, variable in self._kwargs.items():
            if hasattr(variable, 'needs_grad'):
                variable.needs_grad = self.kwarg_flags[key]


class Node:
    """
    Node in the adjoint graph.

    Parameters
    ----------
    op : Operator
        Operator to which the Node refers.
    method : str
        Method within the operator that is to be executed in the adjoint pass.
    idx : int, optional
        Index, within the argument list of the adjoint method, that this node represents.
    nxt : list, optional
        Nodes to which the result of this one will be propagated to.

    """

    def __init__(self, op, method, idx=0, nxt=None):
        self.op_name = op.uname
        self.method = method
        self.idx = idx
        self.next = nxt or []

        if hasattr(op, '_tessera') or \
                (hasattr(op, 'has_tessera') and op.has_tessera and op.is_proxy):
            op = getattr(op, '_tessera')

        self.op = op if self.method != '__noop__' else None

    @property
    def name(self):
        """
        Full name of the node.

        """
        return '%s.%s' % (self.op_name, self.method)

    @property
    def name_idx(self):
        """
        Name of the node including its index.

        """
        return '%s.%s:%d' % (self.op_name, self.method, self.idx)

    def add_next(self, node):
        """
        Add node to the list of next nodes.

        Parameters
        ----------
        node : Node

        Returns
        -------

        """
        nxt = [each.name_idx for each in self.next]

        if node.name_idx not in nxt:
            self.next.append(node)

    def copy(self):
        """
        Create a copy of the node.

        Returns
        -------
        Node

        """
        node = Node(self.op, self.method, self.idx)
        node.next = [each.copy() for each in self.next]

        return node

    def __repr__(self):
        nxt = ', '.join([each.name_idx for each in self.next])
        return '<%s, next:[%s]>' % (self.name, nxt)


class Graph:
    """
    Class representing an adjoint graph.

    """

    def __init__(self):
        self.nodes = OrderedDict()

    def add(self, node):
        """
        Add node to the graph.

        Parameters
        ----------
        node : Node

        Returns
        -------

        """
        if node.name not in self.nodes:
            self.nodes[node.name] = node

        node = self.nodes[node.name]

        for nxt in node.next:
            self.add(nxt)

        return node

    @staticmethod
    def toposort(root):
        """
        Iterate over the graph in topological order, starting a the given
        root and all the way to the leaves.

        Parameters
        ----------
        root : Node

        Returns
        -------
        iterable

        """
        next_counts = dict()
        stack = [root]
        while stack:
            node = stack.pop()

            if node.name in next_counts:
                next_counts[node.name] += 1

            else:
                next_counts[node.name] = 1
                stack.extend(node.next)

        available_nodes = [root]
        while available_nodes:
            node = available_nodes.pop()

            yield node

            for nxt in node.next:
                if next_counts[nxt.name] == 1:
                    available_nodes.append(nxt)

                else:
                    next_counts[nxt.name] -= 1

    def print(self, root=None):
        """
        Print the graph.

        Parameters
        ----------
        root : Node, optional
            Root to start the printing from if topological sorting
            is wanted.

        Returns
        -------

        """
        print(self.__repr__(root))

    def __repr__(self, root=None):
        if root is None:
            nodes = self.nodes.values()
        else:
            nodes = self.toposort(root)

        nodes = ''.join(['\t* ' + str(each) + '\n' for each in nodes])
        return '<graph %s>\n%s' % (id(self), nodes)


class Variable:
    """
    Variables are the inputs and outputs of operators, and track the
    graph through which they have travelled.

    Parameters
    ----------
    name : str, optional
        Name of the varible, defaults to automatic name.
    needs_grad : bool, optional
        Whether or not the gradient wrt to this variable is
        needed, and thus whether or not the adjoint graph starting
        from this variable needs to be constructed, defaults to False.

    """

    _count = 0

    def __init__(self, *args, **kwargs):
        cls = self.__class__
        name = kwargs.pop('name', cls.__name__.lower())
        self._init_name = name

        runtime = mosaic.runtime()
        runtime = runtime.uid if runtime else 'head'

        uname = '%s:%s_%d' % (runtime,
                              cls.__name__.lower(),
                              cls._count)

        self.name = name or uname

        uid = uuid.uuid5(uuid.NAMESPACE_OID, uname).hex
        self.uname = '%s-%d-%s' % (name or cls.__name__.lower(),
                                   cls._count,
                                   uid)

        cls._count += 1

        self.grad = None
        self.prec = None
        self.transform = kwargs.pop('transform', None)

        self.graph = Graph()
        self.prev_op = None
        self.needs_grad = kwargs.pop('needs_grad', False)

    async def adjoint(self, grad=None, **kwargs):
        """
        Run the adjoint graph that has this variable as its root.

        Parameters
        ----------
        grad : optional
            Gradient seed to start the adjoint run.
        kwargs : optional
            Extra arguments to pass on through the adjoint run.

        Returns
        -------

        """
        # init grad
        grad = grad or 1.0

        # no need to run graph
        if self.prev_op is None:
            await self.__call_adjoint__(grad, **kwargs)
            self.clear_graph()
            return self

        runtime = mosaic.runtime()

        def dealloc(objs):
            def _dealloc(*args):
                loop = mosaic.get_event_loop()
                for obj in objs:
                    loop.run(obj.drop)
            return _dealloc

        prev = dict()
        prev[self.prev_op.name_idx] = grad
        returns = []
        parallel_returns = []
        deallocs = []
        for node in self.graph.toposort(self.prev_op):
            kwargs_ = kwargs.copy()

            if node.method == '__noop__':
                continue

            # prepare output grads
            output_names = [each for each in prev.keys() if each.startswith(node.name)]
            output_names.sort()
            output_grads = [prev[each] for each in output_names]

            # call adjoint method
            try:
                method = getattr(node.op, node.method)
            except AttributeError:
                method = getattr(node.op.obj, node.method)
            if hasattr(node.op, 'is_parameter') and node.op.is_parameter:
                ret = method(*output_grads, **{**kwargs_, **{'eager': True}})
            else:
                ret = method(*output_grads, **kwargs_)

            if inspect.iscoroutine(ret) or inspect.iscoroutinefunction(ret):
                ret = await ret

            if isinstance(ret, TaskProxy):
                if len(deallocs):
                    ret.add_done_callback(dealloc(deallocs))

                if (not hasattr(node.op, 'has_tessera') or not node.op.has_tessera or not node.op.is_proxy) and \
                        (not hasattr(node.op, 'is_parameter') or not node.op.is_parameter):
                    returns.append(ret)
                else:
                    parallel_returns.append(ret)

                input_grads = ret.outputs
            else:
                if inspect.iscoroutine(ret) or inspect.iscoroutinefunction(ret):
                    ret = await ret

                input_grads = (ret,) if not isinstance(ret, tuple) else ret

            try:
                if len(input_grads) < len(node.next):
                    raise RuntimeError('Provided %d outputs for the adjoint of operator %s, '
                                       'but %d were expected' % (len(input_grads), node.op.uname, len(node.next)))
            except TypeError:
                pass

            # store gradients for future use
            deallocs = []
            for nxt_index in range(len(node.next)):
                nxt = node.next[nxt_index]
                input_grad = input_grads[nxt_index]

                if nxt.method == '__noop__':
                    continue

                if nxt.name_idx in prev:
                    input_grad = await _maybe_sum(prev[nxt.name_idx], input_grad)

                if not isinstance(input_grad, types.awaitable_types) \
                        and hasattr(nxt.op, 'runtime_id') and nxt.op.runtime_id != runtime.uid:
                    input_grad = await runtime.put(input_grad)
                    deallocs.append(input_grad)

                prev[nxt.name_idx] = input_grad

        eager = not len(returns) or returns[-1]._eager
        if eager:
            await asyncio.gather(*returns)
        else:
            summ_returns = []
            summ_dependencies = []
            for ret in reversed(returns):
                if ret not in summ_dependencies:
                    summ_returns.append(ret)
                    for runtime_deps in ret._dependencies.values():
                        summ_dependencies += list(runtime_deps.values())

            await asyncio.gather(*summ_returns)

        self.clear_graph()

        return self

    def detach(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph.

        Returns
        -------
        Variable
            Detached variable.

        """
        kwargs['name'] = kwargs.pop('name', self._init_name)
        kwargs['needs_grad'] = kwargs.pop('needs_grad', self.needs_grad)
        kwargs['transform'] = kwargs.pop('transform', self.transform)

        if hasattr(self, 'has_tessera') and self.has_tessera:
            cpy = self.__class__.parameter(*args, **kwargs)
        else:
            cpy = self.__class__(*args, **kwargs)

        if self.grad is not None:
            cpy.grad = self.grad.copy()

        if self.prec is not None:
            cpy.prec = self.prec.copy()

        return cpy

    def as_parameter(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph and re-initialised as a parameter.

        Returns
        -------
        Variable
            Detached variable.

        """
        kwargs['name'] = kwargs.pop('name', self._init_name)
        kwargs['needs_grad'] = kwargs.pop('needs_grad', self.needs_grad)
        kwargs['transform'] = kwargs.pop('transform', self.transform)

        cpy = self.__class__.parameter(*args, **kwargs)

        if self.grad is not None:
            cpy.grad = self.grad.copy()

        if self.prec is not None:
            cpy.prec = self.prec.copy()

        return cpy

    def copy(self, *args, **kwargs):
        """
        Create a variable that shares its characteristics with this object.

        The same parameters as those given to ``__init__`` are valid here. Otherwise the
        new object will be configured to be like this one.

        Returns
        -------
        Variable
            Copied variable.

        """
        kwargs['name'] = kwargs.pop('name', self._init_name)
        kwargs['needs_grad'] = kwargs.pop('needs_grad', self.needs_grad)
        kwargs['transform'] = kwargs.pop('transform', self.transform)

        propagate_tessera = kwargs.pop('propagate_tessera', True)

        if propagate_tessera and hasattr(self, 'has_tessera') and self.has_tessera:
            return self.__class__.parameter(*args, **kwargs)
        else:
            return self.__class__(*args, **kwargs)

    def alike(self, *args, **kwargs):
        """
        Alias for a copy.

        """
        kwargs['propagate_tessera'] = kwargs.pop('propagate_tessera', False)
        return self.copy(*args, **kwargs)

    def clear_graph(self):
        """
        Clear the adjoint graph of the variable.

        Returns
        -------

        """
        self.graph = Graph()
        self.prev_op = None

    def clear_grad(self):
        """
        Clear the gradient buffer of the variable.

        Returns
        -------

        """
        raise NotImplementedError('Unimplemented Variable method clear_grad')

    def process_grad(self):
        """
        Process the gradient of the variable for its use.

        Returns
        -------
        object
            Processed gradient

        """
        raise NotImplementedError('Unimplemented Variable method process_grad')

    async def __call_adjoint__(self, grad, **kwargs):
        """
        Adjoint operation of the variable, which accumulates the given
        gradient on the ``Variable.grad`` attribute.

        Parameters
        ----------
        grad : object
            Provided gradient

        Returns
        -------

        """
        if grad is None or not self.needs_grad or self.grad is None:
            return

        grad_data = grad.data if hasattr(grad, 'data') else grad
        is_nan = np.any(np.isnan(grad_data))
        is_inf = np.any(np.isinf(grad_data))

        if is_nan or is_inf:
            msg = 'Nan or inf detected in %s' % self.name

            problem = kwargs.pop('problem', None)
            shot_id = problem.shot.id if problem is not None else kwargs.pop('shot_id', None)
            if shot_id is not None:
                msg = '(ShotID %d) ' % shot_id + msg

            mosaic.logger().warn(msg)
            return

        self.grad += grad

    def __repr__(self):
        return self.name


class Operator:
    """
    Operators represent operations that, when performed on Variables,
    construct an adjoint graph that can then be executed in an adjoint run
    to calculate necessary gradients.

    Parameters
    ----------
    name : str, optional
        Name of the varible, defaults to automatic name.

    """

    _count = 0

    def __init__(self, *args, **kwargs):
        cls = self.__class__
        name = kwargs.pop('name', None)

        runtime = mosaic.runtime()
        runtime = runtime.uid if runtime else 'head'

        uname = '%s:%s_%d' % (runtime,
                              cls.__name__.lower(),
                              cls._count)

        self.name = name or uname

        uid = uuid.uuid5(uuid.NAMESPACE_OID, uname).hex
        self.uname = '%s-%d-%s' % (name or cls.__name__.lower(),
                                   cls._count,
                                   uid)

        cls._count += 1

        self.inputs = None
        self.num_outputs = None

    @abstractmethod
    async def forward(self, *args, **kwargs):
        """
        Method defining the forward behaviour of the operator. This
        method needs to be defined by classes inheriting from the operator.

        The method can take multiple inputs and produce multiple outputs.
        Outputs of this method should be of type Variable.

        Positional and keyword arguments to forward are processed so that
        present variables are tracked.

        This method should not be called directly from user code.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        pass

    @abstractmethod
    async def adjoint(self, *args, **kwargs):
        """
        Method defining the adjoint behaviour of the operator. This
        method needs to be defined by classes inheriting from the operator.

        The method will be called with positional arguments comprised of:
        the gradients of every output of the forward operation, followed by
        the arguments originally given when calling the forward method.

        The adjoint method needs to return a gradient for each of its
        Variable inputs (or None if the variable does not ``needs_grad``).

        This method should not be called directly from user code.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        pass

    async def __call__(self, *args, **kwargs):
        """
        Operators are executed by calling them. The operator will then
        take care of tracking all necessary Variables.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        # process inputs
        needs_grad = False

        next_ops = []

        args, kwargs = await self._process_inputs(*args, **kwargs)

        for arg in args:
            if hasattr(arg, 'needs_grad') and not isinstance(arg, CMDBase):
                needs_grad |= arg.needs_grad

                if arg.needs_grad and arg.prev_op is None:
                    next_ops.append(Node(arg, '__call_adjoint__', 0))
                elif arg.needs_grad:
                    next_ops.append(arg.prev_op)
                else:
                    next_ops.append(Node(arg, '__noop__', 0))

        # for arg in kwargs.values():
        #     if hasattr(arg, 'needs_grad') and not isinstance(arg, CMDBase):
        #         needs_grad |= arg.needs_grad
        #
        #         if arg.needs_grad and arg.prev_op is None:
        #             next_ops.append(Node(arg, '__call_adjoint__', 0))
        #         elif arg.needs_grad:
        #             next_ops.append(arg.prev_op)
        #         else:
        #             next_ops.append(Node(arg, '__noop__', 0))

        self.inputs = (args, kwargs)

        # call forward
        if inspect.iscoroutinefunction(self.forward):
            outputs = await self.forward(*args, **kwargs)
        else:
            outputs = self.forward(*args, **kwargs)
        outputs = (outputs,) if not isinstance(outputs, tuple) else outputs

        # process outputs
        for idx, output in zip(range(len(outputs)), outputs):
            if needs_grad:
                prev_op = Node(self, '__call_adjoint__', idx, next_ops)

                output.graph.add(prev_op)
                output.prev_op = prev_op

            output.needs_grad = needs_grad

        self.num_outputs = len(outputs)
        outputs = outputs if len(outputs) > 1 else outputs[0]

        return outputs

    async def __call_adjoint__(self, *output_grads, **kwargs):
        """
        This method runs the necessary operations to execute the adjoint
        of the operator.

        Parameters
        ----------
        output_grads
        kwargs

        Returns
        -------

        """
        # process inputs
        output_grads, kwargs = await self._process_inputs(*output_grads, **kwargs)

        # call adjoint
        input_args = self.inputs[0]
        input_kwargs = {**self.inputs[1], **kwargs}

        if inspect.iscoroutinefunction(self.adjoint):
            input_grads = await self.adjoint(*output_grads, *input_args, **input_kwargs)
        else:
            input_grads = self.adjoint(*output_grads, *input_args, **input_kwargs)

        # clean up
        self.inputs = None
        self.num_outputs = None

        return input_grads

    async def _process_inputs(self, *args, **kwargs):
        processed_args = []
        processed_kwargs = dict()

        for arg in args:
            if type(arg) in types.awaitable_types:
                await arg

                if isinstance(arg, types.awaitable_types):
                    continue

                arg = await arg.result()

            processed_args.append(arg)

        for key, arg in kwargs.items():
            if type(arg) in types.awaitable_types:
                await arg

                if isinstance(arg, types.awaitable_types):
                    continue

                arg = await arg.result()

            processed_kwargs[key] = arg

        return processed_args, processed_kwargs

    def __repr__(self):
        return self.name
