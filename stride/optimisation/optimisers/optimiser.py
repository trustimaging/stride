
import numpy as np
from abc import ABC, abstractmethod

import mosaic

from ..step_length import LineSearch
from ..pipelines import ProcessGlobalGradient, ProcessModelIteration


__all__ = ['LocalOptimiser']


class LocalOptimiser(ABC):
    """
    Base class for a local optimiser. It takes the value of the gradient and applies
    it to the variable.

    Parameters
    ----------
    variable : Variable
        Variable to which the optimiser refers.
    step_size : float or LineSearch, optional
        Step size for the update, defaults to constant 1.
    process_grad : callable, optional
        Optional processing function to apply on the gradient prior to applying it.
    process_model : callable, optional
        Optional processing function to apply on the model after updating it.
    kwargs
        Extra parameters to be used by the class.

    """

    def __init__(self, variable, **kwargs):
        if not hasattr(variable, 'needs_grad') or not variable.needs_grad:
            raise ValueError('To be optimised, a variable needs to be set with "needs_grad=True"')

        self.variable = variable
        self.step_size = kwargs.pop('step_size', 1.)
        self.test_step_size = kwargs.pop('test_step_size', 1.)
        self.dump_grad = kwargs.pop('dump_grad', False)
        self.dump_prec = kwargs.pop('dump_prec', False)
        self._process_grad = kwargs.pop('process_grad', ProcessGlobalGradient(**kwargs))
        self._process_model = kwargs.pop('process_model', ProcessModelIteration(**kwargs))
        self.reset_block = kwargs.pop('reset_block', False)
        self.reset_iteration = kwargs.pop('reset_iteration', False)

    def clear_grad(self):
        """
        Clear the internal gradient buffers of the variable.

        Returns
        -------

        """
        self.variable.clear_grad()

    async def pre_process(self, grad=None, processed_grad=None, **kwargs):
        """
        Pre-process the variable gradient before using it to take the step.

        Parameters
        ----------
        grad : Data, optional
            Gradient to use for the step, defaults to variable gradient.
        processed_grad : Data, optional
            Processed gradient to use for the step, defaults to processed variable gradient.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        logger = mosaic.logger()
        logger.perf('Updating variable %s,' % self.variable.name)

        if processed_grad is None:
            if grad is None:
                if hasattr(self.variable, 'is_proxy') and self.variable.is_proxy:
                    await self.variable.pull(attr='grad')

                problem = kwargs.pop('problem', None)
                iteration = kwargs.pop('iteration', None)
                dump_grad = kwargs.pop('dump_grad', self.dump_grad)
                dump_prec = kwargs.pop('dump_prec', self.dump_prec)
                if dump_grad and problem is not None:
                    self.variable.grad.dump(path=problem.output_folder,
                                            project_name=problem.name,
                                            parameter='raw_%s' % self.variable.grad.name,
                                            version=iteration.abs_id+1)

                if dump_prec and self.variable.grad.prec is not None and problem is not None:
                    self.variable.grad.prec.dump(path=problem.output_folder,
                                                 project_name=problem.name,
                                                 version=iteration.abs_id+1)

                grad = self.variable.process_grad(**kwargs)

                if dump_grad and problem is not None:
                    grad.dump(path=problem.output_folder,
                              project_name=problem.name,
                              version=iteration.abs_id+1)

            min_dir = np.min(grad.data)
            max_dir = np.max(grad.data)

            logger.perf('\t grad before processing in range [%e, %e]' %
                        (min_dir, max_dir))

            processed_grad = await self._process_grad(grad, variable=self.variable, **kwargs)

        test_step_size = kwargs.pop('test_step_size', self.test_step_size)
        processed_grad.data[:] *= test_step_size

        min_dir = np.min(processed_grad.data)
        max_dir = np.max(processed_grad.data)

        min_var = np.min(self.variable.data)
        max_var = np.max(self.variable.data)

        logger.perf('\t grad after processing in range [%e, %e]' %
                    (min_dir, max_dir))
        logger.perf('\t variable range before update [%e, %e]' %
                    (min_var, max_var))

        return processed_grad

    async def step(self, step_size=None, grad=None, processed_grad=None, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        step_size : float, optional
            Step size to use for this application, defaults to instance step.
        grad : Data, optional
            Gradient to use for the step, defaults to variable gradient.
        processed_grad : Data, optional
            Processed gradient to use for the step, defaults to processed variable gradient.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        logger = mosaic.logger()

        # make copy of variable
        variable_before = self.variable.copy()

        # pre-process gradient to get update direction
        direction = await self.pre_process(grad=grad,
                                           processed_grad=processed_grad,
                                           **kwargs)

        # select test step size
        step_size = self.step_size if step_size is None else step_size
        step_loop = kwargs.pop('step_loop', None)
        if isinstance(step_size, LineSearch):
            await step_size.init_search(
                variable=self.variable,
                direction=direction,
                **kwargs
            )

        # optimal step size search
        while True:
            # find optimal step
            if isinstance(step_size, LineSearch):
                if step_loop is None:
                    next_step = 1.
                    done_search = True
                else:
                    next_step, done_search = await step_size.next_step(
                        variable=self.variable,
                        direction=direction,
                        **kwargs
                    )
            else:
                next_step = step_size
                done_search = True

            if done_search:
                # cap the step if needed
                max_step = kwargs.pop('max_step', None)
                max_step = np.inf if not isinstance(max_step, (int, float)) else max_step

                unclipped_step = next_step

                if next_step > -0.2:  # if bit -ve, still assume grad is right dirn
                    next_step = max(0.1, min(next_step, max_step))
                elif max_step < np.inf and next_step < -max_step * 0.75:  # in general, prevent -ve steps
                    next_step = -max_step * 0.75
                elif next_step < -0.2:
                    next_step = next_step * 0.25

                logger.perf('\t taking final update step of %e [unclipped step of %e]' % (next_step, unclipped_step))
            else:
                logger.perf('\t taking test step of %e in line search' % next_step)

            # restore variable
            self.variable.data[:] = variable_before.data.copy()

            # update variable
            await self.update_variable(next_step, direction)

            # post-process variable after update
            await self.post_process(**kwargs)

            # if done, stop search
            if done_search:
                break

            # calculate loss change
            self.variable.needs_grad = False
            if hasattr(self.variable, 'push'):
                await self.variable.push(attr='needs_grad')
            try:
                await step_loop()
            finally:
                self.variable.needs_grad = True
                if hasattr(self.variable, 'push'):
                    await self.variable.push(attr='needs_grad')

        return self.variable

    async def post_process(self, **kwargs):
        """
        Perform any necessary post-processing of the variable.

        Parameters
        ----------

        Returns
        -------

        """
        processed_variable = await self._process_model(self.variable, **kwargs)
        self.variable.extended_data[:] = processed_variable.extended_data[:]

        min_var = np.min(self.variable.extended_data)
        max_var = np.max(self.variable.extended_data)

        logger = mosaic.logger()
        logger.perf('\t variable range after update [%e, %e]' %
                    (min_var, max_var))

        self.variable.release_grad()

    @abstractmethod
    async def update_variable(self, step_size, direction):
        """

        Parameters
        ----------
        step_size : float
            Step size to use for updating the variable.
        direction : Data
            Direction in which to update the variable.

        Returns
        -------
        Variable
            Updated variable.

        """
        pass

    def reset(self, **kwargs):
        """
        Reset optimiser state along with any stored buffers.

        Parameters
        ----------
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        pass

    def dump(self, *args, **kwargs):
        """
        Dump latest version of the optimiser.

        Parameters
        ----------
        kwargs
            Extra parameters to be used by the method

        Returns
        -------

        """
        self.variable.dump(*args, **kwargs)

    def load(self, *args, **kwargs):
        """
        Load latest version of the optimiser.

        Parameters
        ----------
        kwargs
            Extra parameters to be used by the method

        Returns
        -------

        """
        self.variable.load(*args, **kwargs)
