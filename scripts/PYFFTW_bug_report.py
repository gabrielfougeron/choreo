import numpy as np
import scipy
import itertools
import pyfftw

eps = 1e-13


def compare_scipy_pyfftw(shape_n):
    
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

    pyfft_object.execute()

    print(np.linalg.norm(y_sp - y_pyfftw))
    assert np.linalg.norm(y_sp - y_pyfftw) < eps



n = 66
shape_n = (n,)
compare_scipy_pyfftw(shape_n)


shape_n = (n,2,2)
compare_scipy_pyfftw(shape_n)

