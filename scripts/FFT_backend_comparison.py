import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)


import numpy as np
import scipy
import mkl_fft
import itertools
import pyfftw

import choreo

eps = 1e-13

n = 66
shape_n = (n,)
shape_m = list(shape_n)
shape_m[0] = shape_n[0]//2+1
shape_m = tuple(shape_m)

x = np.zeros(shape_n)

for idx in itertools.product(*[range(i) for i in shape_n]):
    x[idx] = np.random.random()

y_sp = scipy.fft.rfft(x, axis=0)

planner_effort = 'FFTW_ESTIMATE'
nthreads = 1
y_pyfftw = np.zeros(shape_m, dtype=np.complex128)
direction = 'FFTW_FORWARD'
pyfft_object = pyfftw.FFTW(x, y_pyfftw, axes=(0, ), direction=direction, flags=(planner_effort,), threads=nthreads)      

# pyfft_object.execute()
# choreo.cython.object_execute_in_nogil(pyfft_object)
choreo.cython.object_execute_in_nogil_test(pyfft_object)

print(np.linalg.norm(y_sp - y_pyfftw))
assert np.linalg.norm(y_sp - y_pyfftw) < eps
