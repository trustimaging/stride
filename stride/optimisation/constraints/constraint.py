
from abc import abstractmethod


__all__ = ['Constraint']


class Constraint:
    """
    A constraint that introduces prior information by performing a projection of
    a certain variable onto the feasibility set.

    """

    @abstractmethod
    def project(self, variable, **kwargs):
        """
        Apply the projection.

        Parameters
        ----------
        variable : Variable
            Variable to project.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        pass
