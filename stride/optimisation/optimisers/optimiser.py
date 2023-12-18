
import numpy as np
from abc import ABC, abstractmethod

import mosaic

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
        self.dump_grad = kwargs.pop('dump_grad', False)
        self.dump_prec = kwargs.pop('dump_prec', False)
        self._process_grad = kwargs.pop('process_grad', ProcessGlobalGradient(**kwargs))
        self._process_model = kwargs.pop('process_model', ProcessModelIteration(**kwargs))

    def clear_grad(self):
        """
        Clear the internal gradient buffers of the variable.

        Returns
        -------

        """
        self.variable.clear_grad()

    @abstractmethod
    def step(self, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        pass

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
                dump_grad = kwargs.pop('dump_grad', self.dump_grad)
                dump_prec = kwargs.pop('dump_prec', self.dump_prec)
                if dump_grad and problem is not None:
                    self.variable.grad.dump(path=problem.output_folder,
                                            project_name=problem.name,
                                            version=0)

                if dump_prec and self.variable.grad.prec is not None and problem is not None:
                    self.variable.grad.prec.dump(path=problem.output_folder,
                                                 project_name=problem.name,
                                                 version=0)

                grad = self.variable.process_grad(**kwargs)

            min_dir = np.min(grad.data)
            max_dir = np.max(grad.data)

            logger.perf('\t grad before processing in range [%e, %e]' %
                        (min_dir, max_dir))

            processed_grad = await self._process_grad(grad, **kwargs)

        min_dir = np.min(processed_grad.data)
        max_dir = np.max(processed_grad.data)

        min_var = np.min(self.variable.data)
        max_var = np.max(self.variable.data)

        logger.perf('\t grad after processing in range [%e, %e]' %
                    (min_dir, max_dir))
        logger.perf('\t variable range before update [%e, %e]' %
                    (min_var, max_var))

        return processed_grad

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
