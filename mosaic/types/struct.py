
import copy
import pprint


__all__ = ['Struct']


class Struct:
    """
    Structs represent dictionary-like objects that provide some extra features over Python's ``dict``.

    Their internal representation is an ``dict``, a dictionary that maintains the order in which members
    were added to it. Unlike traditional Python dicts, however, both square-bracket and dot notation are allowed
    to access the members of the Struct.

    When a Struct is not ``mutable``, its members can only be assigned once, any further attempts at modifying
    the value of a member will result in an ``AttributeError``. For ``mutable`` Structs the values of its
    members can be changed as many times as needed.

    Parameters
    ----------
    content : dict-like, optional
        Dict-like object to initialise the Struct, defaults to empty.
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

    """

    _allowed_attributes = ['_content', '_mutable']

    def __init__(self, content=None, mutable=True):
        if content is None:
            content = {}

        self.__dict__['_content'] = self._prepare_content(content, mutable)
        self.__dict__['_mutable'] = mutable

    @staticmethod
    def _prepare_content(content, mutable):
        _content = {}
        for key, value in content.items():
            if isinstance(value, dict):
                value = Struct(value, mutable=mutable)

            elif isinstance(value, list):
                value = [Struct(o, mutable=mutable)
                         if isinstance(o, dict) else o for o in value]

            _content[key] = value

        return _content

    def _get(self, item):
        if item in self.__dict__['_content']:
            return self.__dict__['_content'][item]

        else:
            if item in self._allowed_attributes:
                return self.__dict__[item]

        raise AttributeError('The attribute "%s" does not exist in the container' % item)

    def __contains__(self, item):
        return item in self.__dict__['_content']

    def __getattr__(self, item):
        return self._get(item)

    def __getitem__(self, item):
        return self._get(item)

    def __delitem__(self, item):
        self.__dict__['_content'].__delitem__(item)

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
        return Struct(content=copy.deepcopy(self._content), mutable=self._mutable)

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
                if isinstance(value, dict):
                    value = Struct(value, mutable=self._mutable)

                elif isinstance(value, list):
                    value = [Struct(o, mutable=self._mutable)
                             if isinstance(o, dict) else o for o in value]

                self._content[key] = value

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
