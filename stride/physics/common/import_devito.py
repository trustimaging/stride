
from devito import *  # noqa: F401

try:
    from devitopro import *  # noqa: F401
    from devitopro.types.enriched import (DiskHostDevice, DiskHost, DiskDevice,  # noqa: F401
                                          HostDevice, Host, Device)
    pro_available = True

except ImportError:
    pro_available = False
