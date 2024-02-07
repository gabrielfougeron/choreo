from .funs import *
from .funs_new import *
from .funs_serial import *
from .test_blis import *

try:
    from .funs_parallel import *
except:
    pass