
import re


__all__ = ['snake_case', 'camel_case']


def snake_case(name):
    """
    Change case to snake case.

    Parameters
    ----------
    name : str
        String in camelcase format to convert into snake case

    Returns
    -------
    str
        String in snake case

    """
    name = re.sub(r"[\-\.\s]", '_', str(name))
    if not name:
        return name
    return lowercase(name[0]) + re.sub(r"[A-Z]", lambda matched: '_' + lowercase(matched.group(0)), name[1:])


def camel_case(name):
    """
    Change case to camel case.

    Parameters
    ----------
    name : str
         String in snake case format to convert into camelcase

    Returns
    -------
    str
        String in camelcase

    """
    name = re.sub(r"^[\-_\.]", '', str(name))
    if not name:
        return name
    return uppercase(name[0]) + re.sub(r"[\-_\.\s]([a-zA-Z0-9])", lambda matched: uppercase(matched.group(1)), name[1:])


def lowercase(name):
    return str(name).lower()


def uppercase(name):
    return str(name).upper()
