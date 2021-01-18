
import copy
import pprint
from collections import OrderedDict

from .immutable import ImmutableObject
from ..utils import camel_case


__all__ = ['Struct']


class Struct(ImmutableObject):
    """
    Structs represent dictionary-like objects that provide some extra features over Python's ``OrderedDict``.

    Their internal representation is an ``OrderedDict``, a dictionary that maintains the order in which members
    were added to it. Unlike traditional Python dicts, however, both square-bracket and dot notation are allowed
    to access the members of the Struct.

    Structs can be ``extensible`` and ``mutable``, which is determined when they are instantiated and remains
    unchanged for the life of the object.

    When a Struct is not ``mutable``, its members can only be assigned once, any further attempts at modifying
    the value of a member will result in an ``AttributeError``. For ``mutable`` Structs the values of its
    members can be changed as many times as needed.

    Whether or not a Struct is ``extensible`` affects how its members are accessed. When trying to find a member
    of the Struct, a naive search is performed first, assuming that the item exists
    within its internal dictionary.

    If the search fails, and the Struct is not ``extensible`` an ``AttributeError`` is raised. Otherwise,
    a new search starts for members in the dictionary with a similar signature to the requested item.

    If this search fails, an ``AttributeError`` is raised. If if does not fail and a match is found, the match
    is returned wrapped in a function that will evaluate whether the variant exists and is callable.

    Parameters
    ----------
    content : dict-like, optional
        Dict-like object to initialise the Struct, defaults to empty.
    extensible : bool, optional
        Whether or not the Struct should be extensible, defaults to ``False``.
    mutable : bool, optional
        Whether or not the Struct should be mutable, defaults to ``True``.

    Examples
    --------
    Let's create an empty, immutable Struct:

    >>> struct = Struct(mutable=False)
    >>> struct.member = 10
    >>> struct.member
    10
    >>> struct['member']
    10
    >>> struct.member = 20
    AttributeError('The attribute member already exists in the container, this container is not mutable')

    If the container was ``mutable`` instead:

    >>> struct = Struct()
    >>> struct.member = 10
    >>> struct.member
    10
    >>> struct.member = 20
    >>> struct.member
    20

    The same way mutability affects member assignment, extensibility affects how the members of the Struct are
    accessed:

    >>> def function_variant_1():
    >>>     print('Executed variant_1')
    >>>
    >>> def function_variant_2():
    >>>     print('Executed variant_2')
    >>>
    >>> class KlassVariant1:
    >>>     def __init__(self):
    >>>         print('Instantiated Variant1')
    >>>
    >>> class KlassVariant2:
    >>>     def __init__(self):
    >>>         print('Instantiated Variant2')
    >>>
    >>> struct = Struct(extensible=True)
    >>> struct.function_variant_1 = function_variant_1
    >>> struct.function_variant_2 = function_variant_2
    >>> struct.KlassVariant1 = KlassVariant1
    >>> struct.KlassVariant2 = KlassVariant2
    >>>
    >>> struct.function(use='variant_1')
    Executed variant_1
    >>> struct.Klass('Variant2')()
    Instantiated Variant2
    >>> struct.Klass('variant_1')()
    Instantiated Variant1

    """

    _allowed_attributes = ['_content', '_extensible', '_mutable']

    def __init__(self, content=None, extensible=False, mutable=True):
        super(Struct, self).__init__()

        if content is None:
            content = OrderedDict()

        self._content = self._prepare_content(content, extensible, mutable)
        self._extensible = extensible
        self._mutable = mutable

    @staticmethod
    def _prepare_content(content, extensible, mutable):
        _content = OrderedDict()
        for key, value in content.items():
            if isinstance(value, (dict, OrderedDict)):
                value = Struct(value, extensible=extensible, mutable=mutable)

            elif isinstance(value, list) and len(value) and isinstance(value[0], (dict, OrderedDict)):
                for index in range(len(value)):
                    value[index] = Struct(value[index], extensible=extensible, mutable=mutable)

            if isinstance(key, str):
                _key = '_'.join(key.split(' '))

            else:
                _key = key

            _content[_key] = value

        return _content

    def _get(self, item):
        if item in super(ImmutableObject, self).__getattribute__('_content').keys():
            return super(ImmutableObject, self).__getattribute__('_content')[item]

        else:
            if super(ImmutableObject, self).__getattribute__('_extensible') is False:
                raise AttributeError('The attribute "%s" does not exist in the container' % item)

            else:
                exists = False

                for key in super(ImmutableObject, self).__getattribute__('_content').keys():
                    if item in key and key.startswith(item):
                        exists = True
                        break

                if exists:
                    def dynamic_content_wrapper(*args, **options):
                        use = options.pop('use', None)
                        if use is None:
                            if len(args):
                                key = args[0]
                                args = args[1:]
                            else:
                                raise ValueError('When calling automatic interface "%s" of an ExtensibleObject the '
                                                 'variant has to be specified' % item)

                        else:
                            key = use

                        if key is not None:
                            # Check first whether it exists with function syntax
                            content_name = '%s_%s' % (item, key)
                            if content_name in super(ImmutableObject, self).__getattribute__('_content').keys():
                                content = super(ImmutableObject, self).__getattribute__('_content')[content_name]

                                if callable(content):
                                    return content(*args, **options)

                            # Otherwise, check for class syntax
                            content_name = '%s%s' % (item, camel_case(key)) if isinstance(key, str) \
                                else '%s%s' % (item, key)
                            if content_name in super(ImmutableObject, self).__getattribute__('_content').keys():
                                content = super(ImmutableObject, self).__getattribute__('_content')[content_name]

                                if callable(content):
                                    return content

                        raise AttributeError(
                            'No valid interface was found for content "%s" with variant "%s"' % (item, key))

                    return dynamic_content_wrapper

                else:
                    raise AttributeError('The attribute "%s" does not exist in the container' % item)

    def __contains__(self, item):
        if item in super(ImmutableObject, self).__getattribute__('_content').keys():
            return True

        else:
            if super(ImmutableObject, self).__getattribute__('_extensible') is False:
                return False

            else:
                exists = False

                for key in super(ImmutableObject, self).__getattribute__('_content').keys():
                    if item in key and key.startswith(item):
                        exists = True
                        break

                return exists

    def __getattr__(self, item):
        return self._get(item)

    def __getitem__(self, item):
        return self._get(item)

    def get(self, item, default=None):
        """
        Returns an item from the Struct or a default value if it is not found.

        Parameters
        ----------
        item : str
            Name of the item to find
        default : object, optional
            Default value to be returned in case the item is not found, defaults to ``None``

        Returns
        -------

        """
        return self._content.get(item, default)

    def pop(self, item, default=None):
        """
        Returns an item from the Struct and deletes it, or returns a default value if it is not found.

        Parameters
        ----------
        item : str
            Name of the item to find
        default : object, optional
            Default value to be returned in case the item is not found

        Returns
        -------

        """
        return self._content.pop(item, default)

    def _set(self, item, value):
        if item in self._allowed_attributes:
            return super(Struct, self).__setattr__(item, value)

        if item in self._content.keys() and not self._mutable:
            raise AttributeError('The attribute "%s" already exists in the container, '
                                 'this container is not mutable' % item)

        else:
            self._content[item] = value
            return value

    def __setattr__(self, item, value):
        self._set(item, value)

    def __setitem__(self, item, value):
        self._set(item, value)

    def delete(self, item):
        """
        Delete an item from the container using its key.

        Parameters
        ----------
        item : str
            Name of the item to delete.

        Returns
        -------

        """
        if item in self._content.keys() and not self._mutable:
            raise AttributeError('The attribute "%s" cannot be deleted from the container, '
                                 'this container is not mutable' % item)

        else:
            del self._content[item]

    def items(self):
        """
        Returns the list of keys and values in the Struct.

        Returns
        -------
        odict_items
            List of keys and values in the Struct.

        """
        return self._content.items()

    def keys(self):
        """
        Returns the list of keys in the Struct.

        Returns
        -------
        odict_keys
            List of keys in the Struct.

        """
        return self._content.keys()

    def values(self):
        """
        Returns the list of values in the Struct.

        Returns
        -------
        odict_values
            List of values in the Struct.

        """
        return self._content.values()

    def copy(self):
        """
        Returns a deepcopy of the Struct.

        Returns
        -------
        Struct
            Copied Struct

        """
        return Struct(content=copy.deepcopy(self._content), extensible=self._extensible, mutable=self._mutable)

    def update(self, content):
        """
        Updates the Struct with the contents of a dict-like object.

        Parameters
        ----------
        content : dict-like
            Content with which to update the Struct

        Returns
        -------

        """
        for key, value in content.items():
            if key not in self._content.keys():
                if isinstance(value, (dict, OrderedDict)):
                    value = Struct(value, extensible=self._extensible, mutable=self._mutable)

                elif isinstance(value, list) and len(value) and isinstance(value[0], (dict, OrderedDict)):
                    for index in range(len(value)):
                        value[index] = Struct(value[index], extensible=self._extensible, mutable=self._mutable)

                if isinstance(key, str):
                    _key = '_'.join(key.split(' '))

                else:
                    _key = key

                self._content[_key] = value

            else:
                if isinstance(self._content[key], Struct):
                    self._content[key].update(value)

                elif isinstance(self._content[key], list) \
                        and len(self._content[key]) \
                        and isinstance(self._content[key][0], Struct):
                    for index in range(len(value)):
                        self._content[key][index].update(value[index])

                else:
                    self._content[key] = value

    def __str__(self, printer=None):
        return pprint.pformat(self._content)

    __repr__ = __str__
