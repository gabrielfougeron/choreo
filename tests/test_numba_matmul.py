import os

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numba
import numpy as np
import time

numba_kwargs = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}



def Python_matmul(a,b,c):


    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] += a[i,k]*b[k,j]

Numba_matmul = numba.jit(Python_matmul,**numba_kwargs)
Numpy_matmul = lambda a,b,c : a.dot(b,out=c)
Numpy_matmul.__name__ = 'Numpy_matmul'


N = 2
L = 2
M = 2

a = np.random.random((N,L))
b = np.random.random((L,M))

assert a.shape[1] == b.shape[0]



ntests = 1000


backends = [
    # Python_matmul,
    Numba_matmul,
    Numpy_matmul,
]

times = []
for backend in backends:
    # Warmup

    c = np.zeros((a.shape[0],b.shape[1]),dtype=np.float64)
    backend(a,b,c)

    print('')
    print(f'Backend : {backend.__name__}')
    
    tbeg = time.perf_counter()
    for itest in range(ntests):
        backend(a,b,c)
    tend = time.perf_counter()

    the_time = tend-tbeg
    times.append(the_time)
    print(f'Time : {the_time}')
    print(f'Rel time wrt first : { times[0] / the_time }')