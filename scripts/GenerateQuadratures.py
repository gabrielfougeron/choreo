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
import time

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 



n = 2
a, b = choreo.ShiftedGaussLegendre3Term(n)


print(f'Current precision {mpmath.mp.dps}')






w, z = choreo.QuadFrom3Term(a,b,n)

# print(w)
# print('')
# print(z)
# 
# print('')
# print('Difference between Eigen weights and closed form')
# for i in range(n):
# 
#     xi = z[i]
#     phi = choreo.EvalAllFrom3Term(a,b,n,xi)
# 
#     wi = 2 * (1 - xi*xi) / ((n * (phi[n-1]) *  mpmath.factorial(2*(n-1))/(mpmath.mpf(2)**(n-1) * mpmath.factorial((n-1))**2) )**2)
# 
#     print(abs(wi-w[i]))

print('')
print('Evaluation of Lagrange on its zeros')
for i in range(n):

    xi = z[i]
    phi = choreo.EvalAllFrom3Term(a,b,n,xi)
    print(phi[n])



nint = choreo.SafeGLIntOrder(n)
print('')
print(n,nint)


aint, bint = choreo.ShiftedGaussLegendre3Term(nint)
wint, zint = choreo.QuadFrom3Term(aint,bint,nint)



lagint = mpmath.matrix(n,1)

for iint in range(nint):

    lag = choreo.EvalLagrange(a,b,n,z,zint[iint])

    for i in range(n):
        lagint[i] = lagint[i] + wint[iint] * lag[i]


# 
# print('')
# print(lagint-w)
# 
# 
# ButcherA = choreo.ComputeButcherA(a,b,n,z=None,wint=None,zint=None,nint=None)
# 
# ButcherA_np = np.array(ButcherA.tolist(),dtype=np.float64)


# print('')
# 
# print(choreo.a_table_Gauss_2)
# print(choreo.b_table_Gauss_2)
# print(choreo.c_table_Gauss_2)
# 
# 
# print(ButcherA)
# print(w)
# print(z)

Butcher_a_np, Butcher_b_np, Butcher_c_np = choreo.ComputeGaussButcherTables(n)


print(choreo.a_table_Gauss_2 - Butcher_a_np)
print(choreo.b_table_Gauss_2 - Butcher_b_np)
print(choreo.c_table_Gauss_2 - Butcher_c_np)



# print(ButcherA)
# print(w)
# print(z)

imin = 2
imax = 20
for i in range(imin,imax+1):
    tbeg = time.perf_counter()
    Butcher_a_np, Butcher_b_np, Butcher_c_np = choreo.ComputeGaussButcherTables(i)
    tend = time.perf_counter()
    print(f'n = {i}, time = {tend-tbeg}')