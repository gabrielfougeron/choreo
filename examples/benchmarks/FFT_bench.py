"""
Benchmark of FFT algorithms
===========================
"""

# %%
import os
import sys


try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

import matplotlib.pyplot as plt
import numpy as np
import scipy
import math


import choreo


# %%
all_backends = []
all_backends_names = []

all_backends.append('scipy')
all_backends_names.append('scipy')

try:
    import mkl_fft
    all_backends.append(mkl_fft._scipy_fft_backend)
    all_backends_names.append('mkl_fft')

except:
    pass

try:

    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(300000)
    # pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
    pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
    # pyfftw.config.PLANNER_EFFORT = 'FFTW_PATIENT'
    # pyfftw.config.PLANNER_EFFORT = 'FFTW_EXHAUSTIVE'

    all_backends.append(pyfftw.interfaces.scipy_fft)
    all_backends_names.append('pyfftw')

except:
    pass

n_backends = len(all_backends)

dpi = 150

figsize = (1600/dpi, n_backends * 800 / dpi)

fig, axs = plt.subplots(
    nrows = n_backends,
    ncols = 1,
    sharex = True,
    sharey = True,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

def fft(x):
    scipy.fft.fft(x)

def rfft(x):
    scipy.fft.rfft(x)

def dct_I(x):
    scipy.fft.dct(x,1)

def dst_I(x):
    scipy.fft.dst(x,1)

def dct_III(x):
    scipy.fft.dct(x,3)

def dst_III(x):
    scipy.fft.dst(x,3)

def prepare_x(n):
    x = np.random.random(n)
    return [(x, 'x')]


for i_backend, backend in enumerate(all_backends):
  
    timings_folder = os.path.join(__PROJECT_ROOT__,'docs','source','_build','benchmarks_out')

    if not(os.path.isdir(timings_folder)):
        os.makedirs(timings_folder)

    basename = 'FFT_bench_' + all_backends_names[i_backend]
    timings_filename = os.path.join(timings_folder,basename+'.npy')


    scipy.fft.set_global_backend(
        backend = backend   ,
        only = True
    )

    n_repeat = 1

    time_per_test = 0.2

    all_sizes = np.array([4*3*5 * 2**n for n in range(15)])

    all_funs = [
        fft,
        rfft,
        dct_I,
        dst_I,
        dct_III,
        dst_III,
    ]

    all_scalings = np.array([
        1,
        2,
        4,
        4,
        8,
        8,
    ])

    all_times = choreo.benchmark.run_benchmark(
        all_sizes,
        all_funs,
        setup = prepare_x,
        n_repeat = 1,
        time_per_test = 0.2,
        timings_filename = timings_filename,
    )

    choreo.plot_benchmark(
        all_times                               ,
        all_sizes                               ,
        all_funs                                ,
        all_x_scalings = all_scalings           ,
        n_repeat = n_repeat                     ,
        fig = fig                               ,
        ax = axs[i_backend, 0]                  ,
        title = all_backends_names[i_backend]   ,
    )

plt.tight_layout()
plt.show()