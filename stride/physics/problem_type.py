from abc import ABC
import numpy as np

import mosaic

from ..core import Operator
from ..problem.base import Gridded

__all__ = ['ProblemTypeBase']


@mosaic.tessera
class ProblemTypeBase(ABC, Gridded, Operator):
    """
    Problem types encode the physics of the forward and inverse problems that we want
    to solve using Stride. In most cases, these physics will correspond to state and adjoint PDEs
    describing problems of interest.

    A problem type could have multiple implementations, depending on how the physics are solved
    or the techniques that are used to solve them.

    For example, the ``iso_acoustic`` problem corresponds to the second-order isotropic acoustic
    wave equation, which currently has one implementation using the Devito library. This implementation
    is contained within the ``iso_acoustic/devito`` folder.

    ``ProblemTypeBase`` provides a convenient template for other problem types to use, but is not needed. Only
    inheriting from ``Operator`` is needed to generate new physics.

    When inheriting from ``ProblemTypeBase``, classes will have to define a certain interface.

    To solve the state or forward problem:

    - ``before_forward``
    - ``run_forward``
    - ``after_forward``

    and to solve the adjoint problem:

    - ``before_adjoint``
    - ``run_adjoint``
    - ``after_adjoint``

    If the problem type has to provide the gradient for a certain optimisation variable, the
    class will also have to define a set of methods per variable:

    - ``prepare_grad_[variable_name]`` will be called before the adjoint run to prepare the calculation of the gradient.
    - ``init_grad_[variable_name]`` will be called after prepare to initialise any necessary buffers.
    - ``get_grad_[variable_name]`` will be called after the adjoint run to retrieve in the calculated gradients.

    in order for the gradients to be calculated.

    """

    def __init__(self, **kwargs):
        Gridded.__init__(self, **kwargs)
        Operator.__init__(self, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Run the state or forward problem.

        Parameters
        ----------

        Returns
        -------

        """
        pre_str = ''
        problem = kwargs.get('problem', None)
        if problem is not None and hasattr(problem, 'shot_id'):
            pre_str = '(ShotID %d) ' % problem.shot_id

        self.logger.perf('%sPreparing to run state for shot' % pre_str)
        self.before_forward(*args, **kwargs)

        self.logger.perf('%sRunning state equation for shot' % pre_str)
        fw_output = self.run_forward(*args, **kwargs)

        self.logger.perf('%sCompleting state equation run for shot' % pre_str)
        output = self.after_forward(*args, **kwargs)
        self.logger.perf('%sCompleted state equation run for shot' % pre_str)

        return output if output is not None else fw_output

    def adjoint(self, *args, **kwargs):
        """
        Run the adjoint problem.

        Parameters
        ----------

        Returns
        -------

        """
        pre_str = ''
        problem = kwargs.get('problem', None)
        if problem is not None and hasattr(problem, 'shot_id'):
            pre_str = '(ShotID %d) ' % problem.shot_id

        self.logger.perf('%sPreparing to run adjoint for shot' % pre_str)
        self.before_adjoint(*args, **kwargs)

        self.logger.perf('%sRunning adjoint equation for shot' % pre_str)
        ad_output = self.run_adjoint(*args, **kwargs)

        self.logger.perf('%sCompleting adjoint equation run for shot' % pre_str)
        output = self.after_adjoint(*args, **kwargs)
        self.logger.perf('%sCompleted adjoint equation run for shot' % pre_str)

        return output if output is not None else ad_output

    def before_forward(self, *args, **kwargs):
        """
        Prepare the problem type to run the state or forward problem.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError('before_forward has not been implemented '
                                  'for objects of type %s' % self.__class__.__name__)

    def run_forward(self, *args, **kwargs):
        """
        Run the state or forward problem.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError('run_forward has not been implemented '
                                  'for objects of type %s' % self.__class__.__name__)

    def after_forward(self, *args, **kwargs):
        """
        Clean up after the state run and retrieve the outputs.

        Parameters
        ----------

        Returns
        -------
        Traces
            Time traces produced by the state run.

        """
        raise NotImplementedError('after_forward has not been implemented '
                                  'for objects of type %s' % self.__class__.__name__)

    def before_adjoint(self, *args, **kwargs):
        """
        Prepare the problem type to run the adjoint problem.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError('before_adjoint has not been implemented '
                                  'for objects of type %s' % self.__class__.__name__)

    def run_adjoint(self, *args, **kwargs):
        """
        Run the adjoint problem.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError('run_adjoint has not been implemented '
                                  'for objects of type %s' % self.__class__.__name__)

    def after_adjoint(self, *args, **kwargs):
        """
        Clean up after the adjoint run and retrieve the gradients (if needed).

        Parameters
        ----------

        Returns
        -------
        gradient or tuple of gradients
            Gradients wrt to the problem inputs.

        """
        raise NotImplementedError('after_adjoint has not been implemented '
                                  'for objects of type %s' % self.__class__.__name__)

    def prepare_grad(self, *wrt, **kwargs):
        """
        Prepare the problem type to calculate the gradients wrt the inputs.

        Parameters
        ----------
        wrt
            Tuple of variables with respect to which the inversion is running.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        list
            List of update rules (if any) for the gradients of the problem type
            with respect to the inputs

        """
        gradient_update = []

        for variable in wrt:
            if variable is None or not variable.needs_grad:
                continue

            method = getattr(self, 'prepare_grad_' + variable.name, None)

            if method is None:
                raise ValueError('Variable %s not implemented' % variable.name)

            update = method(variable, wrt=wrt, **kwargs)

            if not isinstance(update, tuple):
                update = (update,)

            gradient_update += update

        return gradient_update

    def init_grad(self, *wrt, **kwargs):
        """
        Initialise buffers in the problem type to calculate the gradients wrt the inputs.

        Parameters
        ----------
        wrt
            Tuple of variables with respect to which the inversion is running.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """

        for variable in wrt:
            if variable is None or not variable.needs_grad:
                continue

            method = getattr(self, 'init_grad_' + variable.name, None)

            if method is None:
                raise ValueError('Variable %s not implemented' % variable.name)

            method(variable, wrt=wrt, **kwargs)

    def get_grad(self, *wrt, **kwargs):
        """
        Retrieve the gradients calculated wrt to the inputs.

        Parameters
        ----------
        wrt
            Tuple of variables with respect to which the inversion is running.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        tuple
            Tuple with all the requested gradients

        """
        grads = []

        for variable in wrt:
            if variable is None or not variable.needs_grad:
                grads.append(None)
                continue

            method = getattr(self, 'get_grad_' + variable.name, None)

            if method is None:
                raise ValueError('Variable %s not implemented' % variable.name)

            grad = method(variable, wrt=wrt, **kwargs)

            grad_data = grad.data if hasattr(grad, 'data') else grad
            is_nan = np.any(np.isnan(grad_data))
            is_inf = np.any(np.isinf(grad_data))

            if is_nan or is_inf:
                msg = 'Nan or inf detected in %s' % self.name

                problem = kwargs.pop('problem', None)
                shot_id = problem.shot.id if problem is not None else kwargs.pop('shot_id', None)
                if shot_id is not None:
                    msg = '(ShotID %d) ' % shot_id + msg

                self.logger.warn(msg)
                return

            grads.append(grad)

        return tuple(grads)
