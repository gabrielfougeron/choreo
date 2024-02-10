import os
import sys

import numpy as np
import math as m


__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo

# shape_0 =(2,3) 
# shape_1 = (2,2,2)
# 
# shifts = np.zeros((3),dtype=np.intp)
# shifts[1] = m.prod(shape_0)
# shifts[2] = shifts[1] + m.prod(shape_1)
# 
# buf = np.random.random((shifts[2]))
# arr_0 = buf[shifts[0]:shifts[1]].reshape(shape_0)
# arr_1 = buf[shifts[1]:shifts[2]].reshape(shape_1)
# 
# print(buf)
# print(arr_0)
# print(arr_1)
# 
# 
# shape_0 =(2,3) 
# 
# buf = np.random.random((m.prod(shape_0)))
# arr_0 = buf.reshape(shape_0)
# 
# print(buf)
# print(arr_0)
# 
# choreo.cython.test_blis.create_memview(buf, shape_0[0], shape_0[1])
# 
# shape0 = (2,3)
# 
# buf = np.random.random((m.prod(shape0)))
# 
# choreo.cython.test_blis.create_memview(buf,2,3)

for i in range(1):
    print('a',i)
else:
    print("b",i)