import mpmath
import numpy as np


import os
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import math as m
import numpy as np
import sys


__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 



n = 3
a, b = choreo.ShiftedGaussLegendre3Term(n)
w, x = choreo.QuadFrom3Term(a,b,n)




phi = choreo.EvalAllFrom3Term(a,b,n,x)


exit()







print(w)
print('')
print(x)

# print(np.sqrt(3)/3)


