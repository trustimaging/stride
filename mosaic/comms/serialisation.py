
import pickle5
import cloudpickle


__all__ = ['serialise', 'deserialise']


def pickle5_dumps(data):
    out_band = []
    in_band = pickle5.dumps(data, protocol=5, buffer_callback=out_band.append)
    return in_band, out_band


def pickle5_loads(in_band, out_band):
    out_band = [bytearray(each) for each in out_band]
    return pickle5.loads(in_band, buffers=out_band)


def serialise(data):
    """
    Serialise ``data`` using Pickle protocol 5 as a default and, failing that,
    resort to cloudpickle.

    Parameters
    ----------
    data : object

    Returns
    -------
    bytes
        Pickled object, in-band.
    list
        List of zero-copy buffers, out-of-band.

    """
    try:
        return pickle5_dumps(data)
    except pickle5.PicklingError:
        return cloudpickle.dumps(data), []


def deserialise(in_band, out_band):
    """
    Deserialise using Pickle protocol 5.

    Parameters
    ----------
    in_band : bytes
        Pickled object.
    out_band : list
        List of buffers.

    Returns
    -------
    deserialised object

    """
    return pickle5_loads(in_band, out_band)
