"""
Benchmark of FFT implementations available as scipy backends
============================================================
"""

# %% 
# This benchmark compares execution times of several FFT functions using different scipy backends.
# The plots give the measured execution time of the FFT as a function of the input length.
# The input length is of the form 3 * 5 * 2**i, so as to favor powers of 2 and small divisors.

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
import pyquickbench

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

all_backends = {'scipy' : 'scipy'}

try:
    import mkl_fft
    all_backends['MKL'] = mkl_fft._scipy_fft_backend

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
    
    all_backends['PYFFTW'] = pyfftw.interfaces.scipy_fft

except:
    pass

def setup(n, backend):
    scipy.fft.set_global_backend(
        backend = all_backends[backend]   ,
        only = True
    )
    x = np.random.random(n)
    return {'x': x}

basename = 'FFT_bench_scipy_backends'
timings_filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

all_args = {
    'n' : np.array([4*3*5 * 2**n for n in range(15)])           ,
    'backend' : [backend_name for backend_name in all_backends] ,
}

all_funs = [
    fft     ,
    rfft    ,
    dct_I   ,
    dst_I   ,
    dct_III ,
    dst_III ,
]

all_times = pyquickbench.run_benchmark(
    all_args                    ,
    all_funs                    ,
    setup = setup               ,
    filename = timings_filename ,
    title = 'Absolute timings'  ,
)

plot_intent = {
    'n' : 'points'              ,
    'backend' : 'subplot_grid_y',
}

pyquickbench.plot_benchmark(
    all_times                   ,
    all_args                    ,
    all_funs                    ,
    show = True                 ,
    plot_intent = plot_intent   ,
)

# %%
# 

relative_to_val = {
    'backend' : 'MKL'                   ,
    pyquickbench.fun_ax_name : "fft"    ,
}

pyquickbench.plot_benchmark(
    all_times                   ,
    all_args                    ,
    all_funs                    ,
    show = True                 ,
    plot_intent = plot_intent   ,
    relative_to_val = relative_to_val       ,
    title = 'Relative timings wrt MKL fft'  ,
)
