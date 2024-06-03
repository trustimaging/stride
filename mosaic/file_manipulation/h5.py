
import os
import h5py
import numpy as np
from datetime import datetime

from ..utils.change_case import camel_case
from ..types import Struct


__all__ = ['HDF5', 'file_exists']


_protocol_version = '0.1'


class FilterException(Exception):
    pass


def _abs_filename(filename, path=None):
    if not os.path.isabs(filename):
        filename = os.path.join(path, filename)

    return filename


def _decode_list(str_list):
    for index in range(len(str_list)):
        if isinstance(str_list[index], list):
            str_list[index] = _decode_list(str_list[index])

        else:
            str_list[index] = str_list[index].decode('utf-8')

    return str_list


def write(name, obj, group):
    if isinstance(obj, dict):
        if name != '/':
            sub_group = group.create_group(name)
        else:
            sub_group = group
            sub_group.attrs['protocol'] = _protocol_version
            sub_group.attrs['datetime'] = str(datetime.now())
        sub_group.attrs['is_array'] = False

        for key, value in obj.items():
            write(key, value, sub_group)

    elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        sub_group = group.create_group(name)
        sub_group.attrs['is_array'] = True
        sub_group.attrs['len'] = len(obj)

        for index in range(len(obj)):
            sub_group_name = '%s_%08d' % (name, index)
            write(sub_group_name, obj[index], sub_group)

    else:
        _write_dataset(name, obj, group)


def append(name, obj, group):
    if isinstance(obj, dict):
        if name != '/':
            if name not in group:
                sub_group = group.create_group(name)
                sub_group.attrs['is_array'] = False

            else:
                sub_group = group[name]
        else:
            sub_group = group
            sub_group.attrs['protocol'] = _protocol_version
            sub_group.attrs['datetime'] = str(datetime.now())

        for key, value in obj.items():
            append(key, value, sub_group)

    elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        if name not in group:
            sub_group = group.create_group(name)
            sub_group.attrs['is_array'] = True

        else:
            sub_group = group[name]

        for index in range(len(obj)):
            sub_group_name = '%s_%08d' % (name, index)
            append(sub_group_name, obj[index], sub_group)

    else:
        if name not in group:
            _write_dataset(name, obj, group)


def _write_dataset(name, obj, group):
    if name in group:
        return

    is_bytes = False
    if isinstance(obj, bytes):
        is_bytes = True
        obj = np.void(obj)

    is_none = False
    if obj is None:
        is_none = True
        obj = 'None'

    dataset = group.create_dataset(name, data=obj)
    dataset.attrs['is_ndarray'] = isinstance(obj, np.ndarray)
    dataset.attrs['is_list'] = isinstance(obj, list)
    dataset.attrs['is_tuple'] = isinstance(obj, tuple)
    dataset.attrs['is_str'] = isinstance(obj, str)
    dataset.attrs['is_bytes'] = is_bytes
    dataset.attrs['is_none'] = is_none

    if isinstance(obj, list) and len(obj):
        flat_obj = np.asarray(obj).flatten().tolist()
        dataset.attrs['is_str'] = isinstance(flat_obj[0], str)


def read(obj, lazy=True, filter=None, only=None):
    if isinstance(obj, h5py.Group):
        if filter is None:
            filter = {}

        for key, filter_list in filter.items():
            if key in obj:
                value = _read_dataset(obj[key], lazy=False)
                if value not in filter_list:
                    raise FilterException

        if obj.attrs.get('is_array'):
            data = []
            for key in sorted(obj.keys()):
                if only is not None and key not in only:
                    continue
                try:
                    value = read(obj[key], lazy=lazy, filter=filter)
                except FilterException:
                    continue
                data.append(value)
        else:
            data = {}
            for key in obj.keys():
                if only is not None and key not in only:
                    continue
                try:
                    value = read(obj[key], lazy=lazy, filter=filter)
                except FilterException:
                    continue
                data[key] = value

        return data

    elif isinstance(obj, h5py.Dataset):
        return _read_dataset(obj, lazy=lazy)


def _read_dataset(obj, lazy=True):
    if 'is_none' in obj.attrs and obj.attrs['is_none']:
        return None

    if obj.attrs['is_ndarray']:

        def load():
            return obj[()]

        setattr(obj, 'load', load)

        if lazy is True:
            return obj

        else:
            return obj[()]

    elif 'is_bytes' in obj.attrs and obj.attrs['is_bytes']:
        obj = obj[()].tobytes()

        return obj

    else:
        data = obj[()]

        if obj.attrs['is_str'] and not obj.attrs['is_list']:
            data = data.decode('utf-8')

        elif obj.attrs['is_tuple']:
            data = tuple(data)

        elif obj.attrs['is_list']:
            data = list(data.tolist())

            if obj.attrs['is_str']:
                _decode_list(data)

        else:
            data = data.item()

        return data


class HDF5:
    """
    This class provides an interface to read and write HDF5 files. It can be used by instantiating the
    class on its own,

    >>> file = HDF5(...)
    >>> file.write(...)
    >>> file.close()

    or as a context manager,

    >>> with HDF5(...) as file:
    >>>     file.dump(...)

    If a particular version is given, the filename will be generated without checks. If no version is given,
    the ``path`` will be checked for the latest available version of the file.

    The file will have the form ``<project_name>-<parameter in camelcase><extension>`` for version 0 and
    ``<project_name>-<parameter in camelcase>-<version with width of 5><extension>`` for higher versions.

    Parameters
    ----------
    filename : str
        Full path to a file, instead of a file being formed with version.
    path : str
        Location of the file in the filesystem, defaults to the current working directory.
    project_name : str
        Name of the project, the prefix that all files of the project will have.
    parameter : str
        Parameter that determines which specific type of file to look for.
    version : int, optional
        Integer version of the file, starting at 0. If not given, the last available version will be found.
    extension : str, optional
        File extension, defaults to ``.h5``.
    mode : str
        Mode in which the file will be opened.

    """

    def __init__(self, *args, **kwargs):
        self._mode = kwargs.pop('mode')

        if len(args) > 0:
            filename = args[0]
        else:
            filename = kwargs.pop('filename', None)

        path = kwargs.pop('path', None) or os.getcwd()

        if filename is None:
            project_name = kwargs.pop('project_name', None)
            parameter = kwargs.pop('parameter', None)

            if project_name is None or parameter is None:
                raise RuntimeError('Either filename or project_name and parameter are needed to generate a filename')

            file_parameter = camel_case(parameter)
            version = kwargs.pop('version', None)
            version_start = kwargs.pop('version_start', 0)
            extension = kwargs.pop('extension', '.h5')

            if version is None or version < 0:
                version = version_start
                if version > 0:
                    filename = _abs_filename('%s-%s-%05d%s' % (project_name, file_parameter, version, extension), path)
                else:
                    filename = _abs_filename('%s-%s%s' % (project_name, file_parameter, extension), path)
                while os.path.exists(filename):
                    version += 1
                    filename = _abs_filename('%s-%s-%05d%s' % (project_name, file_parameter, version, extension), path)

                if self._mode.startswith('r'):
                    version -= 1

            if version > 0:
                filename = _abs_filename('%s-%s-%05d%s' % (project_name, file_parameter, version, extension), path)

            else:
                filename = _abs_filename('%s-%s%s' % (project_name, file_parameter, extension), path)

        else:
            filename = _abs_filename(filename, path)

        self._filename = filename
        self._file = h5py.File(self._filename, self._mode, libver='latest')

    @property
    def mode(self):
        return self._mode

    @property
    def filename(self):
        return self._filename

    @property
    def file(self):
        return self._file

    def close(self):
        self._file.close()

    def load(self, lazy=True, filter=None, only=None):
        group = self._file['/']
        description = read(group, lazy=lazy, filter=filter, only=only)
        return Struct(description)

    def dump(self, description):
        write('/', description, self._file)

    def append(self, description):
        append('/', description, self._file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def file_exists(*args, **kwargs):
    """
    Check whether a certain file exists.

    The file will have the form
    ``<project_name>-<parameter in camelcase><extension>`` for version 0 and
    ``<project_name>-<parameter in camelcase>-<version with width of 5><extension>`` for higher versions.

    Parameters
    ----------
    project_name : str
        Name of the project, the prefix that all files of the project will have.
    parameter : str
        Parameter that determines which specific type of file to look for.
    version : int
        Integer version of the file, starting at 0.
    extension : str, optional
        File extension, defaults to ``.h5``.
    folder : str, optional
        Location of the file in the filesystem, defaults to the current folder.

    Returns
    -------
    bool
        Whether or not a file of the specified version exists.

    """

    if len(args) > 0:
        filename = args[0]
    else:
        filename = kwargs.pop('filename', None)

    path = kwargs.pop('path', None) or os.getcwd()

    if filename is None:
        project_name = kwargs.pop('project_name', None)
        parameter = kwargs.pop('parameter', None)

        if project_name is None or parameter is None:
            raise RuntimeError('Either filename or project_name and parameter are needed to generate a filename')

        file_parameter = camel_case(parameter)
        version = kwargs.pop('version', None)
        extension = kwargs.pop('extension', '.h5')

        if version > 0:
            filename = _abs_filename('%s-%s-%05d%s' % (project_name, file_parameter, version, extension), path)

        else:
            filename = _abs_filename('%s-%s%s' % (project_name, file_parameter, extension), path)

    filename = _abs_filename(filename, path)

    return os.path.exists(filename)


def rm(*args, **kwargs):
    """
    Remove file.

    The file will have the form
    ``<project_name>-<parameter in camelcase><extension>`` for version 0 and
    ``<project_name>-<parameter in camelcase>-<version with width of 5><extension>`` for higher versions.

    Parameters
    ----------
    project_name : str
        Name of the project, the prefix that all files of the project will have.
    parameter : str
        Parameter that determines which specific type of file to look for.
    version : int
        Integer version of the file, starting at 0.
    extension : str, optional
        File extension, defaults to ``.h5``.
    folder : str, optional
        Location of the file in the filesystem, defaults to the current folder.

    Returns
    -------

    """

    if len(args) > 0:
        filename = args[0]
    else:
        filename = kwargs.pop('filename', None)

    path = kwargs.pop('path', None) or os.getcwd()

    if filename is None:
        project_name = kwargs.pop('project_name', None)
        parameter = kwargs.pop('parameter', None)

        if project_name is None or parameter is None:
            raise RuntimeError('Either filename or project_name and parameter are needed to generate a filename')

        file_parameter = camel_case(parameter)
        version = kwargs.pop('version', None)
        extension = kwargs.pop('extension', '.h5')

        if version > 0:
            filename = _abs_filename('%s-%s-%05d%s' % (project_name, file_parameter, version, extension), path)

        else:
            filename = _abs_filename('%s-%s%s' % (project_name, file_parameter, extension), path)

    filename = _abs_filename(filename, path)

    os.remove(filename)
