
from devito import *

try:
    from devitopro import *
    pro_available = True
except ImportError:
    pro_available = False
