"""
Benchmark of FFT algorithms
===========================
"""

# %% 
# This benchmark compares execution times of several FFT functions using different implementations.
# The plots give the measured execution time of the FFT as a function of the input length. 
# The input length is of the form 3 * 5 * 2**i, so as to favor powers of 2 and small divisors.

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
import scipy
import choreo

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

def setup_all(fft_type, nthreads, all_sizes):

    def scipy_fft(x):
        getattr(scipy.fft, fft_type)(x, workers = nthreads)

    def numpy_fft(x):
        getattr(np.fft, fft_type)(x)

    all_funs = [
        numpy_fft,
        scipy_fft,
    ]
    all_names = [
        "numpy",
        "scipy",
    ]
        
    try:

        import pyfftw
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(300000)

        # planner_effort = 'FFTW_ESTIMATE'
        # planner_effort = 'FFTW_MEASURE'
        # planner_effort = 'FFTW_PATIENT'
        planner_effort = 'FFTW_EXHAUSTIVE'

        pyfftw.config.NUM_THREADS = nthreads
        pyfftw.config.PLANNER_EFFORT = planner_effort

        def rfft_pyfftw_interface(x):
            getattr(pyfftw.interfaces.scipy_fftpack, fft_type)(x, threads = nthreads)

        all_funs.append(rfft_pyfftw_interface)
        all_names.append("pyfftw_interface")
        
        all_builders = {}
        for i in range(len(all_sizes)):
            
            n = all_sizes[i]
            if fft_type in ['fft']:
                x = np.random.random(n) + 1j*np.random.random(n)
            elif fft_type in ['rfft']:
                x = np.random.random(n)
            else:
                raise ValueError(f'No prepare function for {fft_type}')
            
            fft_object = getattr(pyfftw.builders, fft_type)(x, threads = nthreads)
            all_builders[n] = fft_object
        
        def prebuilt_fftw(x):
            builder = all_builders[x.shape[0]]
            builder(x)
        
        all_funs.append(prebuilt_fftw)
        all_names.append("pyfftw_prebuilt")
        
        
        all_custom = {}
        for i in range(len(all_sizes)):
            
            n = all_sizes[i]
            if fft_type in ['fft']:
                x = np.random.random(n) + 1j*np.random.random(n)
                y = np.random.random(n) + 1j*np.random.random(n)
                direction = 'FFTW_FORWARD'
            elif fft_type in ['rfft']:
                x = np.random.random(n)
                m = n//2 + 1
                y = np.random.random(m) + 1j*np.random.random(m)
                direction = 'FFTW_FORWARD'
            else:
                raise ValueError(f'No prepare function for {fft_type}')
            
        
            fft_object = pyfftw.FFTW(x, y, axes=(0, ), direction=direction, flags=(planner_effort,), threads=nthreads, planning_timelimit=None)

            all_custom[n] = fft_object
        
        def custom_fftw(x):
            custom = all_custom[x.shape[0]]
            custom(x)
        
        all_funs.append(custom_fftw)
        all_names.append("pyfftw_custom")
        
    except Exception as ex:
        print(ex)

    try:
        
        if (nthreads == (multiprocessing.cpu_count()//2)):
            
            # This f***** will always run with the maximum available number of threads
            import mkl_fft
            
            def rfft_mkl(x):
                getattr(mkl_fft, fft_type)(x)
            
            all_funs.append(rfft_mkl)
            all_names.append('mkl')

    except:
        pass


    return all_funs, all_names

def plot_all(relative_to=None):

    all_fft_types = [
        'fft',
        'rfft',
    ]

    all_nthreads = set([1,multiprocessing.cpu_count()//2])
    all_sizes = np.array([4*3*5 * 2**n for n in range(15)])

    n_plots = len(all_nthreads) * len(all_fft_types)

    dpi = 150

    figsize = (1600/dpi, n_plots * 800 / dpi)

    fig, axs = plt.subplots(
        nrows = n_plots,
        ncols = 1,
        sharex = True,
        sharey = True,
        figsize = figsize,
        dpi = dpi   ,
        squeeze = False,
    )


    # sphinx_gallery_defer_figures

    for iplot, (nthreads, fft_type) in enumerate(itertools.product(all_nthreads, all_fft_types)):

        all_funs, all_names = setup_all(fft_type, nthreads, all_sizes)
            
        if fft_type in ['fft']:
            def prepare_x(n):
                x = np.random.random(n) + 1j*np.random.random(n)
                return [(x, 'x')]
            
        elif fft_type in ['rfft']:
            def prepare_x(n):
                x = np.random.random(n)
                return [(x, 'x')]
        else:
            raise ValueError(f'No prepare function for {fft_type}')

        plural = 's' if nthreads == 1 else ''

        basename = f'PYFFT_bench_{fft_type}_{nthreads}_thread{plural}'
        timings_filename = os.path.join(timings_folder,basename+'.npy')

        all_times = choreo.benchmark.run_benchmark(
            all_sizes                       ,
            all_funs                        ,
            setup = prepare_x               ,
            filename = timings_filename     ,
        )

        choreo.plot_benchmark(
            all_times           ,
            all_sizes           ,
            all_funs            ,
            all_names           ,
            fig = fig           ,
            ax = axs[iplot,0]   ,
            title = f'{fft_type} on {nthreads} thread{plural}'    ,
            relative_to = relative_to   ,
        )
            
        plt.tight_layout()
    plt.show()


# sphinx_gallery_end_ignore
# %%
plot_all(relative_to=None)

# %%
plot_all(relative_to='scipy')

