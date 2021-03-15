
from abc import ABC, abstractmethod


__all__ = ['ProblemTypeBase']


class ProblemTypeBase(ABC):
    """
    Problem types encode the physics of the forward and inverse problems that we want
    to solve using Stride. In most cases, these physics will correspond to state and adjoint PDEs
    describing problems of interest.

    A problem type could have multiple implementations, depending on how the physics are solved
    or the techniques that are used to solve them.

    For example, the ``acoustic`` problem corresponds to the second-order isotropic acoustic
    wave equation, which currently has one implementation using the Devito library. This implementation
    is contained within the ``acoustic/devito`` folder.

    Problem types inherit from this base class, and have to comply with a certain interface by
    defining, at least, a series of methods.

    To solve the state or forward problem:

    - ``before_state``
    - ``state``
    - ``after_state``

    and to solve the adjoint problem:

    - ``before_adjoint``
    - ``adjoint``
    - ``after_adjoint``

    If the problem type has to provide the gradient for a certain optimisation variable, the
    class will also have to define a pair of methods per variable:

    - ``set_grad_[variable_name]`` will be called before the adjoint run to prepare the calculation of the gradient.
    - ``get_grad_[variable_name]`` will be called after the adjoint run to fill in the calculated gradients.

    in order for the gradients to be calculated.

    """

    space_order = -1
    time_order = -1

    def __init__(self):
        self._problem = None

        self._state_operator = None
        self._adjoint_operator = None

    def set_problem(self, problem, **kwargs):
        """
        Set up the problem or sub-problem that needs to be run.

        Parameters
        ----------
        problem : SubProblem or Problem
            Problem on which the physics will be executed
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        self._problem = problem

    @abstractmethod
    def before_state(self, save_wavefield=False, **kwargs):
        """
        Prepare the problem type to run the state or forward problem.

        Parameters
        ----------
        save_wavefield : bool, optional
            Whether or not the wavefield needs to be stored, defaults to False.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        pass

    @abstractmethod
    def state(self, **kwargs):
        """
        Run the state or forward problem.

        Parameters
        ----------
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        pass

    @abstractmethod
    def after_state(self, save_wavefield=False, **kwargs):
        """
        Clean up after the state run and retrieve the time traces.

        If requested, also provide a saved wavefield.

        Parameters
        ----------
        save_wavefield : bool, optional
            Whether or not the wavefield needs to be stored, defaults to False.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Traces
            Time traces produced by the state run.
        Data or None
            Wavefield produced by the state run, if any.

        """
        pass

    @abstractmethod
    def before_adjoint(self, wrt, adjoint_source, wavefield, **kwargs):
        """
        Prepare the problem type to run the adjoint problem.

        Parameters
        ----------
        wrt : VariableList
            List of variables for which the inverse problem is being solved.
        adjoint_source : Traces
            Adjoint source to use in the adjoint propagation.
        wavefield : Data
            Stored wavefield from the forward run, to use as needed.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        pass

    @abstractmethod
    def adjoint(self, wrt, adjoint_source, wavefield, **kwargs):
        """
        Run the adjoint problem.

        Parameters
        ----------
        wrt : VariableList
            List of variables for which the inverse problem is being solved.
        adjoint_source : Traces
            Adjoint source to use in the adjoint propagation.
        wavefield : Data
            Stored wavefield from the forward run, to use as needed.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        pass

    @abstractmethod
    def after_adjoint(self, wrt, adjoint_source, wavefield, **kwargs):
        """
        Clean up after the adjoint run and retrieve the time gradients (if needed).

        Parameters
        ----------
        wrt : VariableList
            List of variables for which the inverse problem is being solved.
        adjoint_source : Traces
            Adjoint source to use in the adjoint propagation.
        wavefield : Data
            Stored wavefield from the forward run, to use as needed.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        VariableList
            Updated variable list with gradients added to them.

        """
        pass

    def set_grad(self, wrt, **kwargs):
        """
        Prepare the problem type to calculate the gradients wrt the inputs.

        Parameters
        ----------
        wrt : VariableList
            List of variable with respect to which the inversion is running.
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
            method = getattr(self, 'set_grad_' + variable.name, None)

            if method is None:
                raise ValueError('Variable %s not implemented' % variable.name)

            update = method(variable, **kwargs)
            gradient_update += update

        return gradient_update

    def get_grad(self, wrt, **kwargs):
        """
        Retrieve the gradients calculated wrt to the inputs.

        Parameters
        ----------
        wrt : VariableList
            List of variable with respect to which the inversion is running.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        VariableList
            Updated variable list with gradients added to them.

        """
        for variable in wrt:
            method = getattr(self, 'get_grad_' + variable.name, None)

            if method is None:
                raise ValueError('Variable %s not implemented' % variable.name)

            method(variable, **kwargs)

        return wrt
