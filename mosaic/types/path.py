

__all__ = ['Path']


class Path(str):
    """
    A Path is a ``str`` that contains meta-information about the filesystem path it represents.

    It does not have much functionality at the moment, but will be enriched progressively.
    """

    def __init__(self, *args, **kwargs):
        str.__init__(*args, **kwargs)
        self.is_path = True
