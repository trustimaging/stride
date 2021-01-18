
import sys

from ..utils import camel_case


__all__ = ['ExtensibleObject', 'extensible_module']


class ExtensibleObject:
    """
    Similarly to the ``extensible`` option in Struct, a class that inherits from ExtensibleObject
    provides a means for accessing variants of its attributes and methods without having to use an ``if-else``
    construct.

    Examples
    --------
    >>> class Klass(ExtensibleObject):
    >>>     class KlassVariant1:
    >>>         def __init__(self):
    >>>             print('Instantiated Variant1')
    >>>
    >>>     class KlassVariant2:
    >>>         def __init__(self):
    >>>             print('Instantiated Variant2')
    >>>
    >>>     def function_variant_1(self):
    >>>         print('Executed variant_1')
    >>>
    >>>     def function_variant_2(self):
    >>>         print('Executed variant_2')
    >>>
    >>> klass = Klass()
    >>> klass.function(use='variant_1')
    Executed variant_1
    >>> klass.Klass('Variant2')()
    Instantiated Variant2
    >>> klass.Klass('variant_1')()
    Instantiated Variant1
    """

    def __getattribute__(self, item):
        try:
            return super(ExtensibleObject, self).__getattribute__(item)

        except AttributeError:
            exists = False
            __class__ = super(ExtensibleObject, self).__getattribute__('__class__')

            for key in dir(self):
                if item in key and key.startswith(item):
                    exists = True
                    break

            if exists:
                def dynamic_method_wrapper(*args, **options):
                    use = options.pop('use', False)
                    if not use:
                        if len(args):
                            key = args[0]
                            args = args[1:]
                        else:
                            raise ValueError('When calling automatic interface "%s" of an ExtensibleObject the '
                                             'variant has to be specified' % item)

                    else:
                        key = use

                    if key is not None:
                        method_name = '%s_%s' % (item, key)
                        if method_name in dir(self):
                            method = super(ExtensibleObject, self).__getattribute__(method_name)

                            if callable(method):
                                return method(*args, **options)

                    raise AttributeError('No valid interface was found for method "%s" with variant "%s"' % (item, key))

                return dynamic_method_wrapper

            else:
                raise


def extensible_module(name):
    """
    This function extends a module to provide it with extensibility, as defined for
    Struct and extensible object.

    TODO - Document this feature.

    Parameters
    ----------
    name : str
        Name of the module.

    Returns
    -------
    callable

    """

    def get_attribute(item):
        module = sys.modules[name]
        exists = False
        for key in dir(module):
            if item in key and key.startswith(item):
                exists = True
                break

        if exists:
            def dynamic_method_wrapper(*args, **options):
                use = options.pop('use', False)
                if not use:
                    if len(args):
                        key = args[0]
                        args = args[1:]
                    else:
                        raise ValueError('When calling automatic interface "%s" of an extensible module the '
                                         'variant has to be specified' % item)

                else:
                    key = use

                if key is not None:
                    # Check first whether it exists with function syntax
                    content_name = '%s_%s' % (item, key)
                    if content_name in dir(module):
                        content = getattr(module, content_name)

                        if callable(content):
                            return content(*args, **options)

                    # Otherwise, check for class syntax
                    content_name = '%s%s' % (item, camel_case(key)) if isinstance(key, str) else '%s%s' % (item, key)
                    if content_name in dir(module):
                        content = getattr(module, content_name)

                        if callable(content):
                            return content

                raise AttributeError('No valid interface was found for method "%s" with variant "%s"' % (item, key))

            return dynamic_method_wrapper

        else:
            raise AttributeError('module %s has no attribute %s' % (name, item))

    return get_attribute
