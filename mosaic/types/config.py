
import copy
from collections import OrderedDict

from .struct import Struct


__all__ = ['Config']


class Config(Struct):
    """
    A Config is a specific type of (``mutable`` and not ``extensible``) Struct that populates itself
    with some provided default values. This allows to have some default configuration that can then be
    superseded by the user.

    Parameters
    ----------
    content : dict-like, optional
        Dict-like object to initialise the Struct, defaults to empty.
    defaults : dict-like, optional
        Dict-like object that provides the default values of the Config object.
    """

    _allowed_attributes = ['_content', '_extensible', '_mutable', '_defaults']

    def __init__(self, content=None, defaults=None):
        super(Config, self).__init__(mutable=True, extensible=False)

        if defaults is None:
            defaults = OrderedDict()

        if content is None:
            content = OrderedDict()

        self._defaults = defaults

        defaulted_content = copy.deepcopy(defaults)
        defaulted_content.update(content)
        self._content.update(defaulted_content)

    @property
    def defaults(self):
        """
        Access the default values of the Config object.

        Returns
        -------
        OrderedDict
            Dictionary containing the default values of this Config.

        """
        return self._defaults

    def copy(self):
        """
        Returns a deepcopy of the Config.

        Returns
        -------
        Config
            Copied Config

        """
        return Config(content=copy.deepcopy(self._content), defaults=copy.deepcopy(self._defaults))
