"""
Benchmark of inplace conjugation of arrays
==========================================
"""

# %% 
#
# This is a benchmark of different ways to perform inplace conjugation of a complex numpy array.

# sphinx_gallery_start_ignore

import os
import sys
import multiprocessing
import itertools

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

import matplotlib.pyplot as plt
import numpy as np
import numba as nb

numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

all_sizes = np.array([2**n for n in range(25)])

dpi = 150

figsize = (1600/dpi,  800 / dpi)

fig, ax = plt.subplots(
    figsize = figsize,
    dpi = dpi   ,
)

# ForceBenchmark = True
ForceBenchmark = False

all_funs = []

# sphinx_gallery_end_ignore

def numpy_ufunc_outofplace(x):
    y = np.conjugate(x)
    
def numpy_ufunc_inplace(x):
    np.conjugate(x, out=x)
    
def numpy_inplace_mul(x):
    x.imag *= -1

def numpy_subs(x):
    x.imag = -x.imag 
    
@nb.jit("void(complex128[::1])", **numba_opt_dict)
def numba_loop_typedef(x):
    
    for i in range(x.shape[0]):
        x.imag[i] = -x.imag[i]
        
@nb.jit(**numba_opt_dict)
def numba_loop(x):
    
    for i in range(x.shape[0]):
        x.imag[i] = -x.imag[i]
    
@nb.jit(**numba_opt_dict, parallel=True)
def numba_loop_parallel(x):
    
    for i in nb.prange(x.shape[0]):
        x.imag[i] = -x.imag[i]
    
@nb.vectorize(['complex128(complex128)'], nopython=True, cache=True, target='cpu')
def numba_vectorize(x):
    return x.real - 1j*x.imag

@nb.vectorize(['complex128(complex128)'], nopython=True, cache=True, target='cpu')
def numba_vectorize_conj(x):
    return x.conjugate()

@nb.vectorize(['complex128(complex128)'], nopython=True, cache=True, target='parallel')
def numba_vectorize_conj_parallel(x):
    return x.conjugate()
    
# sphinx_gallery_start_ignore
    
all_funs = [
    numpy_ufunc_outofplace ,
    numpy_ufunc_inplace ,
    numpy_inplace_mul ,
    numpy_subs ,
    numba_loop_typedef ,
    numba_loop ,
    numba_loop_parallel ,
    numba_vectorize ,
    numba_vectorize_conj ,
    numba_vectorize_conj_parallel ,
]

def prepare_x(n):
    x = np.random.random(n) + 1j*np.random.random(n)
    return {'x': x}
    
basename = f'Inplace_conjugation_bench'
timings_filename = os.path.join(timings_folder,basename+'.npy')

all_times = pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    setup = prepare_x               ,
    filename = timings_filename     ,
    ForceBenchmark = ForceBenchmark     ,
)

pyquickbench.plot_benchmark(
    all_times           ,
    all_sizes           ,
    all_funs            ,
    fig = fig           ,
    ax = ax             ,
    title = f'Inplace conjugation'    ,
)
    
plt.tight_layout()

# sphinx_gallery_end_ignore

plt.show()
