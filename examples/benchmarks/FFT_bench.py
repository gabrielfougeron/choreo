"""
Benchmark of FFT algorithms
===========================
"""

# %% 
# This benchmark compares execution times of several FFT functions using different backends
# The plots give the measured execution time of the FFT as a function of the input length

# sphinx_gallery_start_ignore

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
import choreo

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def fft(x):
    scipy.fft.fft(x)

def rfft(x):
    scipy.fft.rfft(x)

def dct_I(x):
    N = x.shape[0]
    n = N // 2 + 1
    scipy.fft.dct(x[:n],1)

def dst_I(x):
    N = x.shape[0]
    n = N // 2 - 1
    scipy.fft.dst(x[:n],1)

def dct_III(x):
    N = x.shape[0]
    n = N // 4
    scipy.fft.dct(x[:n],3)

def dst_III(x):
    N = x.shape[0]
    n = N // 4
    scipy.fft.dst(x[:n],3)
    
# sphinx_gallery_start_ignore

def prepare_x(n):
    x = np.random.random(n)
    return [(x, 'x')]


all_backends = []
all_backends_names = []

all_backends.append('scipy')
all_backends_names.append('Scipy backend')

try:
    import mkl_fft
    all_backends.append(mkl_fft._scipy_fft_backend)
    all_backends_names.append('MKL backend')

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
    all_backends_names.append('PYFFTW backend')

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


for i_backend, backend in enumerate(all_backends):
  
    basename = 'FFT_bench_' + all_backends_names[i_backend].replace(' ','_')
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

    all_times = choreo.benchmark.run_benchmark(
        all_sizes,
        all_funs,
        setup = prepare_x,
        n_repeat = 1,
        time_per_test = 0.2,
        filename = timings_filename,
    )

    choreo.plot_benchmark(
        all_times                               ,
        all_sizes                               ,
        all_funs                                ,
        n_repeat = n_repeat                     ,
        fig = fig                               ,
        ax = axs[i_backend, 0]                  ,
        title = all_backends_names[i_backend]   ,
    )
    
plt.tight_layout()
plt.show()

# sphinx_gallery_end_ignore
