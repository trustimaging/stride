

__all__ = ['Variable']


class Variable:

    def __new__(cls, variable):
        instance = variable.copy()

        instance.grad = variable.alike(instance.name+'_grad')
        instance.prec = variable.alike(instance.name+'_prec')

        return instance
