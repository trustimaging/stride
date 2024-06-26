
import random
import numpy as np
import functools
import contextlib
import pickle


__all__ = ['maybe_compress', 'decompress']


try:
    import blosc

    n = blosc.set_nthreads(6)
    if hasattr('blosc', 'releasegil'):
        blosc.set_releasegil(True)
except ImportError:
    blosc = False


def identity(data):
    return data


compression_methods = {None: {'compress': identity, 'decompress': identity}}
compression_methods[False] = compression_methods[None]  # alias

default_compression = None


with contextlib.suppress(ImportError):
    import zlib

    compression_methods['zlib'] = {'compress': zlib.compress, 'decompress': zlib.decompress}

with contextlib.suppress(ImportError):
    import snappy

    def _fixed_snappy_decompress(data):
        # snappy.decompress() doesn't accept memoryviews
        if isinstance(data, (memoryview, bytearray)):
            data = bytes(data)
        return snappy.decompress(data)

    compression_methods['snappy'] = {
        'compress': snappy.compress,
        'decompress': _fixed_snappy_decompress,
    }
    default_compression = 'snappy'

with contextlib.suppress(ImportError):
    import lz4

    try:
        # try using the new lz4 API
        import lz4.block

        lz4_compress = lz4.block.compress
        lz4_decompress = lz4.block.decompress
    except ImportError:
        # fall back to old one
        lz4_compress = lz4.LZ4_compress
        lz4_decompress = lz4.LZ4_uncompress

    # helper to bypass missing memoryview support in current lz4
    # (fixed in later versions)

    def _fixed_lz4_compress(data):
        try:
            return lz4_compress(data)
        except TypeError:
            if isinstance(data, (memoryview, bytearray)):
                return lz4_compress(bytes(data))
            else:
                raise

    def _fixed_lz4_decompress(data):
        try:
            return lz4_decompress(data)
        except (ValueError, TypeError):
            if isinstance(data, (memoryview, bytearray)):
                return lz4_decompress(bytes(data))
            else:
                raise

    compression_methods['lz4'] = {
        'compress': _fixed_lz4_compress,
        'decompress': _fixed_lz4_decompress,
    }
    default_compression = 'lz4'


with contextlib.suppress(ImportError):
    import zstandard

    zstd_compressor = zstandard.ZstdCompressor(
        level=22,
        threads=6,
    )

    zstd_decompressor = zstandard.ZstdDecompressor()

    def zstd_compress(data):
        return zstd_compressor.compress(data)

    def zstd_decompress(data):
        return zstd_decompressor.decompress(data)

    compression_methods['zstd'] = {
        'compress': zstd_compress,
        'decompress': zstd_decompress
    }
    default_compression = 'zstd'


with contextlib.suppress(ImportError):
    import blosc

    compression_methods['blosc'] = {
        'compress': functools.partial(blosc.compress, clevel=5, cname='lz4'),
        'decompress': functools.partial(blosc.decompress, as_bytearray=True),
    }
    default_compression = 'blosc'


user_compression = 'auto'
if user_compression != 'auto':
    if user_compression in compression_methods:
        default_compression = user_compression
    else:
        raise ValueError(
            'Default compression "%s" not found.\n'
            'Choices include auto, %s'
            % (user_compression, ', '.join(sorted(map(str, compression_methods))))
        )


def ensure_bytes(s):
    """
    Attempt to turn `s` into bytes.

    Parameters
    ----------
    s : Any
        The object to be converted. Will correctly handled
        * str
        * bytes
        * objects implementing the buffer protocol (memoryview, ndarray, etc.)

    Returns
    -------
    b : bytes

    Raises
    ------
    TypeError
        When `s` cannot be converted

    Examples
    --------
    >>> ensure_bytes('123')
    b'123'
    >>> ensure_bytes(b'123')
    b'123'
    """
    if isinstance(s, bytes):
        return s
    elif hasattr(s, 'encode'):
        return s.encode()
    else:
        try:
            return bytes(s)
        except Exception as e:
            raise TypeError('Object %s is neither a bytes object nor has an encode method' % s) from e


def byte_sample(b, size, n):
    """
    Sample a bytestring from many locations

    Parameters
    ----------
    b : bytes or memoryview
    size : int
        size of each sample to collect
    n : int
        number of samples to collect
    """

    if type(b) is memoryview:
        b = memoryview(np.asarray(b).ravel())

    if type(b) is np.ndarray:
        b = b.reshape(-1)

    starts = [random.randint(0, len(b) - size) for j in range(n)]
    ends = []
    for i, start in enumerate(starts[:-1]):
        ends.append(min(start + size, starts[i + 1]))
    ends.append(starts[-1] + size)

    parts = [b[start:end] for start, end in zip(starts, ends)]
    return b''.join(map(ensure_bytes, parts))


def maybe_compress(payload, min_size=1e4, sample_size=1e4, nsamples=5):
    """
    Maybe compress payload:

    1.  We don't compress small messages
    2.  We sample the payload in a few spots, compress that, and if it doesn't
        do any good we return the original
    3.  We then compress the full original, it it doesn't compress well then we
        return the original
    4.  We return the compressed result

    """
    if isinstance(payload, pickle.PickleBuffer):
        payload = memoryview(payload)

    if type(payload) is memoryview or hasattr(payload, 'nbytes'):
        nbytes = payload.nbytes
    else:
        nbytes = len(payload)

    if not default_compression:
        return None, payload
    if nbytes < min_size:
        return None, payload
    if nbytes > 2e9:  # Too large, compression libraries often fail
        return None, payload

    min_size = int(min_size)
    sample_size = int(sample_size)

    compression = default_compression
    compress = compression_methods[default_compression]['compress']

    # Compress a sample, return original if not very compressed, but not for memoryviews
    if type(payload) is not memoryview:
        sample = byte_sample(payload, sample_size, nsamples)
        if len(compress(sample)) > 0.9 * len(sample):  # sample not very compressible
            return None, payload

    if default_compression and blosc and type(payload) is memoryview:
        # Blosc does itemsize-aware shuffling, resulting in better compression
        compressed = blosc.compress(payload,
                                    typesize=payload.itemsize,
                                    cname='lz4',
                                    clevel=5)
        compression = 'blosc'
    else:
        compressed = compress(ensure_bytes(payload))

    if len(compressed) > 0.9 * nbytes:  # full data not very compressible
        return None, payload
    else:
        return compression, compressed


def decompress(compression, payload):
    """
    Decompress payload according to information in the header

    """
    return compression_methods[compression]['decompress'](payload)
