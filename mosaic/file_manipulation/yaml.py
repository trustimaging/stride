
import os
import re
import yaml
from yaml import Loader
from collections import OrderedDict

from ..types import Path, Struct, ImportedFunction


__all__ = ['parse_yaml', 'parse_python']


# Add a constructor to avoid !tags from being parsed
def default_constructor(loader, tag_suffix, node):
    return tag_suffix + ' ' + node.value


yaml.add_multi_constructor('', default_constructor)

# Make sure scientific notation without dot or exponent sign is understood
Loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def _parse_dict(config, work_dir):
    for key, value in config.items():
        if isinstance(value, str):
            if value.startswith('!func'):
                value = value[5:].split(',')

                if len(value) > 1:
                    function_name = value[0].strip()
                    module_name = value[1].strip()

                else:
                    function_name = value[0].strip()
                    module_name = value[0].strip()

                if not module_name.startswith('/'):
                    module_name = os.path.join(work_dir, module_name)
                module_name = os.path.abspath(module_name)

                function_path = module_name if module_name.endswith('.py') else '%s.py' % module_name
                function_dir = os.path.dirname(function_path)
                module_name = os.path.basename(module_name)

                value = ImportedFunction(function_name, module_name, (function_dir, function_path,))

            elif value.startswith('!eval'):
                value = eval(value[5:])

            elif value.startswith('!path'):
                value = Path(os.path.abspath(os.path.join(work_dir, value[5:].strip())))

        elif isinstance(value, dict):
            value = _parse_dict(value, work_dir)

        config[key] = value

    return config


def parse_yaml(file_path):
    """
    This utility function uses the default YAML constructors to parse most of the content of the files.

    It also includes constructors to handle our custom YAML tags ``!func``, ``!eval`` and ``!path``.

    Parameters
    ----------
    file_path : str
        Absolute path to the YAML file to be loaded

    Returns
    -------
    OrderedDict
        Ordered nested dictionary with the parsed contents of the YAML file

    Examples
    --------
    The custom tags that are have been implemented allow for the definition of new YAML types. The ``!func``
    declares that a certain field is a callable, importable function by using the syntax:

    .. code-block:: python

        field_name: !func function_name

    This syntax will automatically look for a function called ``function_name`` inside a file named
    ``function_name.py``. The paths of the files are automatically resolved with respect to the path of the YAML file.
    This syntax can be further expanded by specifying a file where the function is located:

    .. code-block:: python

        field_name: !func function_name, path_to_file/file_name

    This syntax will result in the import of a function called ``function_name`` inside a file named
    ``path_to_file/file_name.py``.
    The ``!eval`` tag allows for the evaluation of valid one-line python expressions:

    .. code-block:: python

        field_name: !eval 2*4

    will be evaluated to the integer 8. More complex expressions could also be created:

    .. code-block:: python

        field_name: !eval [2, 5, 7, 8]

    This syntax should also be used to assign tuple values, that YAML does not handle otherwise:

    .. code-block:: python

        field_name: !eval (2, 3)

    The ``!path`` tag should be used to specify fields that are meant to represent paths in the filesystem. Whether
    absolute or relative, they will be automatically resolved with respect to the location of the YAML file:

    .. code-block:: python

        field_name: !path ./folder

    will be evaluated to ``/path/of/yaml/file/folder``.
    """
    work_dir = os.path.dirname(file_path)
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=Loader)
        config = _parse_dict(config, work_dir)

    return config


def parse_python(obj, folder=None, comments=None, level=0):
    """
    This function generates a YAML-like string out of a Python object, usually a dictionary or a Struct.

    It takes care of handling the format of the files so they do not look cluttered, it handles out custom tags
    ``!func``, ``!eval`` and ``!path``, and allows for the inclusion of comments in the parsed string.

    Parameters
    ----------
    obj : any Python literal, Path or Struct
        Python object to parse into a string, usually a dictionary
    folder : str, optional
        Folder with respect to which the paths will be resolved
    comments : dict, optional
        A dictionary mimicking the structure of the input dictionary that contains comments to be added to the
        parsed lines
    level : int, optional
        Nesting level of the object being parsed, usually 0

    Returns
    -------
    str
        String containing a parsed version of the Python input

    """
    if comments is None:
        comments = {}

    if callable(obj):
        rel_path, ext = os.path.splitext(os.path.relpath(obj.__globals__['__file__'], folder))
        if ext == '.py':
            parsed = '!func %s, %s' % (obj.__name__, rel_path)
        else:
            parsed = '!func %s, %s' % (obj.__name__, rel_path + ext)

    elif isinstance(obj, tuple):
        parsed = '!eval ('
        for each in obj:
            if isinstance(each, str):
                parsed += '\'%s\', ' % each
            else:
                parsed += '%s, ' % parse_python(each, folder=folder)

        parsed += ')'

    elif isinstance(obj, list):
        parsed = '!eval ['
        for each in obj:
            if isinstance(each, str):
                parsed += '\'%s\', ' % each
            else:
                parsed += '%s, ' % parse_python(each, folder=folder)

        parsed += ']'

    elif isinstance(obj, Path):
        rel_path, ext = os.path.splitext(os.path.relpath(obj, folder))
        if ext == '.py':
            parsed = '!path %s' % rel_path
        else:
            parsed = '!path %s' % rel_path + ext

    elif isinstance(obj, float):
        if obj/1e4 > 1 or obj < 1e-2:
            parsed = '%e' % obj

            parsed = parsed.split('e')
            parsed[0] = parsed[0].rstrip('0').rstrip('.')
            parsed = 'e'.join(parsed)
        else:
            parsed = '%g' % obj

    elif isinstance(obj, int):
        parsed = '%d' % obj

    elif isinstance(obj, (dict, Struct)):
        _obj = OrderedDict()
        for key, value in obj.items():
            _obj[key] = parse_python(value, folder=folder, comments=comments.get(key, {}), level=level+1)

        parsed = '''\
        '''

        for key, value in _obj.items():
            if isinstance(obj[key], (dict, Struct)):
                parsed += '\n'

            if isinstance(comments.get(key, ''), str):
                _comment = comments.get(key, '')
            else:
                _comment = ''

            parsed += f'''\
\n{'    '*level}{key} : {value}    {_comment}'''

    else:
        parsed = obj

    return parsed
