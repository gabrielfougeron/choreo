import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)


import numpy as np
import scipy
import mkl_fft
import itertools

shape = (2,2,1)

x = np.zeros(shape)

for idx in itertools.product(*[range(i) for i in shape]):
    x[idx] = np.random.random()

y_sp = scipy.fft.rfft(x, axis=0, n=2*shape[0])
y_mkl = mkl_fft._numpy_fft.rfft(x, axis=0, n=2*shape[0])

# 
# print(y_sp)
# print(y_mkl)

print(np.linalg.norm(y_sp - y_mkl))