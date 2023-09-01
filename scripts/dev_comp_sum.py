import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)


import numpy as np
import math as m
import scipy
import choreo
import time

n = int(1e5)

a = np.zeros((n),dtype=np.float64)
x = -10
choreo.scipy_plus.cython.test.inplace_taylor_poly(a,x)

# a = np.random.random(n)

exact_res = m.fsum(a)

# np.random.shuffle(a)

kmax = 5

tbeg = time.perf_counter()
res = choreo.scipy_plus.cython.eft_lib.naive_sum_vect(a)
err = abs(exact_res-res)/exact_res
tend = time.perf_counter()
print(f'naive_sum_vect      : {err:.2e} in {tend-tbeg:.5f} s')

tbeg = time.perf_counter()
res = sum(a)
err = abs(exact_res-res)/exact_res
tend = time.perf_counter()
print(f'sum                 : {err:.2e} in {tend-tbeg:.5f} s')

tbeg = time.perf_counter()
res = np.sum(a)
err = abs(exact_res-res)/exact_res
tend = time.perf_counter()
print(f'np.sum              : {err:.2e} in {tend-tbeg:.5f} s')

tbeg = time.perf_counter()
res = m.fsum(a)
err = abs(exact_res-res)/exact_res
tend = time.perf_counter()
print(f'm.fsum              : {err:.2e} in {tend-tbeg:.5f} s')

for k in range(1,kmax):

    tbeg = time.perf_counter()
    res = choreo.scipy_plus.cython.eft_lib.SumK(a, k)
    err = abs(exact_res-res)/exact_res
    tend = time.perf_counter()
    print(f'SumK {k}              : {err:.2e} in {tend-tbeg:.5f} s')
    
for k in range(1,kmax):

    tbeg = time.perf_counter()
    res = choreo.scipy_plus.cython.eft_lib.FastSumK(a, k)
    err = abs(exact_res-res)/exact_res
    tend = time.perf_counter()
    print(f'FastSumK {k}          : {err:.2e} in {tend-tbeg:.5f} s')







