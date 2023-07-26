"""
Benchmark of FFT algorithms
===========================
"""

# %%
import os
import choreo
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

timings_folder = os.path.join(__PROJECT_ROOT__,'docs','source','_build','benchmarks_out')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

basename = 'FFT_bench'
timings_filename = os.path.join(timings_folder,basename+'.npy')

# %%

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

# %%
n_repeat = 1

time_per_test = 0.2

all_sizes = [4*3*5 * 2**n for n in range(12)]

all_funs = [
    fft,
    # rfft,
    # dct_I,
    # dst_I,
    # dct_III,
    # dst_III,
]


all_times = choreo.benchmark.run_benchmark(
    all_sizes,
    all_funs,
    setup = prepare_x,
    n_repeat = 1,
    time_per_test = 0.2,
    timings_filename = timings_filename,
    show = True,
)

