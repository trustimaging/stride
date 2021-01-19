

__all__ = ['Path']


class Path(str):
    """
    A Path is a ``str`` that contains meta-information about the filesystem path it represents.

    It does not have much functionality at the moment, and might be dropped at some point.
    """

    def __init__(self, *args, **kwargs):
        str.__init__(*args, **kwargs)
        self.is_path = True
