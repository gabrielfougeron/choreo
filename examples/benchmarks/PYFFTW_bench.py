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
# 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

import multiprocessing
import itertools
import tqdm

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

import matplotlib.pyplot as plt
import numpy as np
import scipy

import choreo
DP_Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.txt")
choreo.find.Load_wisdom_file(DP_Wisdom_file)


base_shape = [4,4]
def make_shape(n, axis):
    shape = base_shape.copy()
    shape[axis] = n
    return shape

# import mkl_fft
# scipy.fft.set_global_backend(
#     backend = mkl_fft._scipy_fft_backend   ,
#     only = True
# )

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

def setup_all(fft_type, nthreads, all_sizes):
    
    all_n = all_sizes['n']
    all_axis = all_sizes['axis']

    def scipy_fft(x, axis):
        getattr(scipy.fft, fft_type)(x, axis=axis, workers = nthreads, overwrite_x = True)

    def numpy_fft(x, axis):
        getattr(np.fft, fft_type)(x, axis=axis)

    all_funs = {
        "numpy" : numpy_fft,
        "scipy" : scipy_fft,
    }
        
    try:

        import pyfftw
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(300000)

        # planner_effort = 'FFTW_ESTIMATE'
        # planner_effort = 'FFTW_MEASURE'
        planner_effort = 'FFTW_PATIENT'
        # planner_effort = 'FFTW_EXHAUSTIVE'

        pyfftw.config.NUM_THREADS = nthreads
        pyfftw.config.PLANNER_EFFORT = planner_effort
        
        print('Planning FFTW')
        all_custom = {}
        for i in tqdm.tqdm(range(len(all_n))):
            for j in range(len(all_axis)):
                
                n = all_n[i]
                axis = all_axis[j]
                
                if fft_type in ['fft']:
                    
                    shape_in = tuple(make_shape(n, axis))
                    x = pyfftw.empty_aligned(shape_in, dtype=np.complex128)
                    y = pyfftw.empty_aligned(shape_in, dtype=np.complex128)
                    direction = 'FFTW_FORWARD'
                elif fft_type in ['rfft']:
                    
                    m = n//2 + 1
                    shape_in = tuple(make_shape(n, axis))
                    shape_out = tuple(make_shape(m, axis))
                    
                    x = pyfftw.empty_aligned(shape_in, dtype=np.float64)
                    y = pyfftw.empty_aligned(shape_out, dtype=np.complex128)
                    direction = 'FFTW_FORWARD'
                elif fft_type in ['irfft']:
                    
                    m = n//2 + 1
                    shape_in = tuple(make_shape(m, axis))
                    shape_out = tuple(make_shape(n, axis))
                    
                    x = pyfftw.empty_aligned(shape_in, dtype=np.complex128)
                    y = pyfftw.empty_aligned(shape_out, dtype=np.float64)
                    direction = 'FFTW_BACKWARD'
                else:
                    raise ValueError(f'No prepare function for {fft_type}')
                
                fft_object = pyfftw.FFTW(x, y, axes=(axis, ), direction=direction, flags=(planner_effort, 'FFTW_DESTROY_INPUT'), threads=nthreads, planning_timelimit=None)      

                all_custom[shape_in] = fft_object
            
        def custom_fftw(x, axis):
            custom = all_custom[x.shape]
            custom()
        
        all_funs["pyfftw_custom"] = custom_fftw
        
    except Exception as ex:
        print(ex)

    try:
        
        # if (nthreads == (multiprocessing.cpu_count()//2)):
        #     
        #     # This f***** will always run with the maximum available number of threads
        #     import mkl_fft
        #     
        #     def rfft_mkl(x):
        #         getattr(mkl_fft, fft_type)(x)
        #     
        #     all_funs['mkl'] = rfft_mkl
            
        # This f***** will always run with the maximum available number of threads
        import mkl_fft
        
        def rfft_mkl(x, axis):
            # getattr(mkl_fft, fft_type)(x)
            getattr(mkl_fft._numpy_fft, fft_type)(x, axis=axis)
        
        all_funs['mkl'] = rfft_mkl

    except:
        pass


    return all_funs

def plot_all(relative_to_val = None):

    all_fft_types = [
        'fft',
        'rfft',
        'irfft',
    ]

    all_nthreads = [
        1, 
        # multiprocessing.cpu_count()//2
    ]
    
    all_sizes = {
        "n" : np.array([3* 2**n for n in range(14)]),
        "axis" : [0,1],
    }

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

        all_funs = setup_all(fft_type, nthreads, all_sizes)
            
        if fft_type in ['fft']:
            def prepare_x(n, axis):
                shape = make_shape(n, axis)
                x = np.random.random(shape) + 1j*np.random.random(shape)
                return {'x': x, 'axis':axis}
            
        elif fft_type in ['rfft']:
            def prepare_x(n, axis):
                shape = make_shape(n, axis)
                x = np.random.random(shape)
                return {'x': x, 'axis':axis}            
        elif fft_type in ['irfft']:
            def prepare_x(n, axis):
                m=n//2+1
                shape = make_shape(m, axis)
                x = np.random.random(shape)
                return {'x': x, 'axis':axis}
        else:
            raise ValueError(f'No prepare function for {fft_type}')

        plural = 's' if nthreads > 1 else ''

        basename = f'PYFFT_bench_{fft_type}_{nthreads}_thread{plural}'
        timings_filename = os.path.join(timings_folder,basename+'.npz')
        
        n_repeat = 10

        all_times = pyquickbench.run_benchmark(
            all_sizes                       ,
            all_funs                        ,
            n_repeat = n_repeat             ,
            setup = prepare_x               ,
            filename = timings_filename     ,
            ShowProgress=True               ,
            # ForceBenchmark=True             ,
        )
            
        plot_intent = {
            "n" : 'points'                           ,
            "axis" : 'curve_linestyle'                  ,
            pyquickbench.fun_ax_name :  'curve_color'  ,
            pyquickbench.repeat_ax_name :  'reduction_min'  ,
            # pyquickbench.repeat_ax_name :  'reduction_avg'  ,
        }

        pyquickbench.plot_benchmark(
            all_times           ,
            all_sizes           ,
            all_funs            ,
            fig = fig           ,
            ax = axs[iplot,0]   ,
            title = f'{fft_type} on {nthreads} thread{plural}'    ,
            relative_to_val = relative_to_val,
            plot_intent = plot_intent,  
        )
            
        plt.tight_layout()
    plt.show()


# sphinx_gallery_end_ignore
# %%
plot_all()

# %%

relative_to_val = {
    pyquickbench.fun_ax_name:'scipy',
    # 'axis':0,
}

plot_all(relative_to_val=relative_to_val)


choreo.find.Write_wisdom_file(DP_Wisdom_file)
