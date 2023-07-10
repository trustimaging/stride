
from devito import *  # noqa: F401

try:
    from devitopro import *  # noqa: F401
    pro_available = True
except ImportError:
    pro_available = False
