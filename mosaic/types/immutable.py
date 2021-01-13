
import inspect


__all__ = ['ImmutableObject']


class ImmutableObject:
    """
    A class that inherits from ImmutableObject produces the same effect as a class with all attributes
    private in C++ programming, i.e. the attributes and methods of the instances of this class cannot be
    changed by code external to the class.

    Examples
    --------
    >>> class Klass(ImmutableObject):
    >>>     def __init__(self):
    >>>         self.attribute = 10
    >>>
    >>>     def change_attr(self):
    >>>         self.attribute = 20
    >>>
    >>> klass = Klass()
    >>> klass.attribute
    10
    >>> klass.attribute = 30
    AttributeError('Objects of class "Klass" are immutable')
    >>> klass.change_attr()
    >>> klass.attribute
    20
    """

    def __setattr__(self, key, value):
        calling_frame = inspect.stack()[1][0]
        args, _, _, value_dict = inspect.getargvalues(calling_frame)

        calling_class = None
        if len(args) and args[0] == 'self':
            instance = value_dict.get('self', None)
            if instance is not None:
                calling_class = getattr(instance, '__class__', None)

        if calling_class is not None and calling_class == self.__class__:
            super(ImmutableObject, self).__setattr__(key, value)

        else:
            raise AttributeError('Objects of class "%s" are immutable' % self.__class__)
