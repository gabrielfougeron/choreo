import os
import threadpoolctl

import scipy
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))


import numpy as np
import choreo
Sym = choreo.ActionSym.Random(nbody = 10, geodim = 4)
print(Sym.signature.ActionSym == Sym)
