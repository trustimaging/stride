
from devito import *  # noqa: F401
from devito.types import Symbol, Scalar  # noqa: F401
from devito.symbolics import INT, IntDiv, CondEq  # noqa: F401
from devito import TimeFunction as TimeFunctionOSS  # noqa: F401

try:
    from devitopro import *  # noqa: F401
    from devitopro.types.enriched import (DiskHostDevice, DiskHost, DiskDevice,  # noqa: F401
                                          HostDevice, Host, Device, Disk, NoLayers)
    from devitopro.types.compressed import CompressedTimeFunction
    pro_available = True

except ImportError:
    DiskHostDevice = None
    DiskHost = None
    DiskDevice = None
    HostDevice = None
    Host = None
    Device = None
    Disk = None
    NoLayers = None
    pro_available = False
    CompressedTimeFunction = None
