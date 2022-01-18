
import zict
import struct

from ..utils import sizeof
from ..comms.serialisation import serialise, deserialise
from ..comms.compression import maybe_compress, decompress


__all__ = ['SpillBuffer']


def nbytes(frame, _bytes_like=(bytes, bytearray)):
    """
    Secure funtion to get number of bytes.

    """
    if isinstance(frame, _bytes_like):
        return len(frame)
    else:
        try:
            return frame.nbytes
        except AttributeError:
            return len(frame)


def len_frames(frames):
    """
    Process frames to generate length header.

    """
    num_frames = len(frames)
    nbytes_frames = map(nbytes, frames)

    return struct.pack(f'Q{num_frames}Q', num_frames, *nbytes_frames)


def unlen_frames(data):
    """
    Process bytes to extract multipart frames.

    """
    data = memoryview(data)

    fmt_size = struct.calcsize('Q')

    (num_frames,) = struct.unpack_from('Q', data)
    lengths = struct.unpack_from(f'{num_frames}Q', data, fmt_size)

    frames = []
    start = fmt_size * (1 + num_frames)
    for length in lengths:
        end = start + length

        frames.append(data[start:end])

        start = end

    return frames


def serialise_and_compress(data):
    """
    Wrapper for serialisation and (maybe) compression.

    """
    data = serialise(data)

    compression = []
    compressed_data = []

    _compression, _compressed_data = maybe_compress(data[0])
    compression.append(_compression)
    compressed_data.append(_compressed_data)

    if len(data[1]) > 0:
        _compression, _compressed_data = zip(*map(maybe_compress, data[1]))
        compression.append(_compression)
        compressed_data.append(_compressed_data)

    else:
        compression.append([])
        compressed_data.append([])

    header = {
        'compression': compression,
    }

    header = serialise(header)[0]

    multipart_data = [header]
    multipart_data += [compressed_data[0]]
    multipart_data += compressed_data[1]

    multipart_data.insert(0, len_frames(multipart_data))

    return multipart_data


def decompress_and_deserialise(multipart_data):
    """
    Wrapper for deserialisation and decompression.

    """
    multipart_data = unlen_frames(multipart_data)

    header = deserialise(multipart_data[0], [])

    if len(multipart_data) > 2:
        compressed_data = [multipart_data[1], multipart_data[2:]]
    else:
        compressed_data = [multipart_data[1], []]

    data = []

    _data = decompress(header['compression'][0], compressed_data[0])
    data.append(_data)

    _data = [decompress(compression, payload)
             for compression, payload in zip(header['compression'][1], compressed_data[1])]
    data.append(_data)

    data = deserialise(data[0], data[1])

    return data


class SpillBuffer(zict.Buffer):
    """
    MutableMapping that automatically spills out key/value pairs to disk when
    the total size of the stored data exceeds the target.

    """

    def __init__(self, spill_directory, target):
        self.spilled_by_key = {}
        self.spilled_total = 0

        storage = zict.Func(serialise_and_compress,
                            decompress_and_deserialise,
                            zict.File(spill_directory, mode='w'))

        super().__init__(dict(), storage, target,
                         weight=self._weight,
                         fast_to_slow_callbacks=[self._on_evict],
                         slow_to_fast_callbacks=[self._on_retrieve])

    @property
    def memory(self):
        """
        Key/value pairs stored in RAM. Alias of zict.Buffer.fast.

        """
        return self.fast

    @property
    def disk(self):
        """
        Key/value pairs spilled out to disk. Alias of zict.Buffer.slow.

        """
        return self.slow

    @staticmethod
    def _weight(key, value):
        return sizeof(value)

    def _on_evict(self, key, value):
        b = sizeof(value)

        self.spilled_by_key[key] = b
        self.spilled_total += b

    def _on_retrieve(self, key, value):
        self.spilled_total -= self.spilled_by_key.pop(key)

    def __setitem__(self, key, value):
        self.spilled_total -= self.spilled_by_key.pop(key, 0)

        super().__setitem__(key, value)

        if key in self.slow:
            # value is individually larger than target so it went directly to slow.
            # _on_evict was not called.
            b = sizeof(value)

            self.spilled_by_key[key] = b
            self.spilled_total += b

    def __delitem__(self, key):
        self.spilled_total -= self.spilled_by_key.pop(key, 0)
        super().__delitem__(key)
