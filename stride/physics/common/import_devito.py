
from devito import *  # noqa: F401

try:
    from devitopro import *  # noqa: F401
    pro_available = True

    from devitopro.types.enriched import (DiskHostDevice, DiskHost, DiskDevice,
                                          HostDevice, Host, Device)

except ImportError:
    pro_available = False
