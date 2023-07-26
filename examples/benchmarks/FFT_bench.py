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
from mkl_fft import _scipy_fft_backend
from pyfftw.interfaces import scipy_fft

import choreo


all_backends = [
    'scipy',
    _scipy_fft_backend,
    scipy_fft
]

all_backends_names = [
    'scipy',
    'mkl_fft',
    'pyfftw',
]

n_backends = len(all_backends)

figsize = (8,9)

fig, axs = plt.subplots(
    nrows = n_backends,
    ncols = 1,
    sharex = True,
    sharey = True,
    figsize = figsize,
)


for i_backend, backend in enumerate(all_backends):
  
    scipy.fft.set_backend(
        backend = backend   ,
        only = True
    )

    timings_folder = os.path.join(__PROJECT_ROOT__,'docs','source','_build','benchmarks_out')

    if not(os.path.isdir(timings_folder)):
        os.makedirs(timings_folder)

    basename = 'FFT_bench_' + all_backends_names[i_backend]
    timings_filename = os.path.join(timings_folder,basename+'.npy')

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
        ax = axs[i_backend]                     ,
        title = all_backends_names[i_backend]   ,
    )

plt.show()