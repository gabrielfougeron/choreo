
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import functools
import time
import matplotlib.pyplot as plt
import numpy as np
import math as m

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 
import scipy
import numba

ndim = 1
x_span = (0., 1.)
method = 'Gauss'
nsteps = 1
quad = choreo.scipy_plus.SegmQuad.ComputeQuadrature(method, nsteps)

# print(quad)
# all_quad = dir(quad)
# for item in all_quad:
#     print(item)


print(f'{quad.w = }')
print(f'{quad.x = }')
print(f'{quad.th_cvg_rate = }')