import numpy as np
import math as m

shape_0 =(2,3) 
shape_1 = (2,2,2)

shifts = np.zeros((3),dtype=np.intp)
shifts[1] = m.prod(shape_0)
shifts[2] = shifts[1] + m.prod(shape_1)

buf = np.random.random((shifts[2]))
arr_0 = buf[shifts[0]:shifts[1]].reshape(shape_0)
arr_1 = buf[shifts[1]:shifts[2]].reshape(shape_1)

print(buf)
print(arr_0)
print(arr_1)