
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
    try:
        return pickle5_dumps(data)
    except pickle5.PicklingError:
        return cloudpickle.dumps(data), []


def deserialise(in_band, out_band):
    return pickle5_loads(in_band, out_band)
