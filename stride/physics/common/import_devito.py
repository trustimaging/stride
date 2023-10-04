
from devito import *  # noqa: F401

try:
    from devitopro import *  # noqa: F401
    from devitopro.types.enriched import (DiskHostDevice, DiskHost, DiskDevice,  # noqa: F401
                                          HostDevice, Host, Device, NoLayers)
    pro_available = True

except ImportError:
    DiskHostDevice = None
    DiskHost = None
    DiskDevice = None
    HostDevice = None
    Host = None
    Device = None
    NoLayers = None
    pro_available = False
