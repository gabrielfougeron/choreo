"""

"""

import os
import sys
import matplotlib.pyplot as plt
import math
import threadpoolctl

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

threadpoolctl.threadpool_limits(limits=1).__enter__()

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
bench_filename = os.path.join(bench_folder, 'Kepler.npz')

# %%

import numpy as np
import choreo
import pyquickbench

def kepler_error(nint_fac, eccentricity):
    
    nbody = 2
    mass = 1
    
    NBS, segmpos = choreo.NBodySyst.KeplerEllipse(nbody, nint_fac, mass, eccentricity)
    # NBS.fft_backend = "fftw"
    
    params_buf = NBS.segmpos_to_params(segmpos)
    action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)

    res = np.array([
        np.linalg.norm(action_grad, ord = np.inf)                   ,
        np.linalg.norm(action_grad, ord = 2) / action_grad.shape[0] ,
        np.linalg.norm(action_grad, ord = 1) / action_grad.shape[0] ,
    ])
    
    return res

    
def setup(nint_fac, eccentricity):
    return {"nint_fac":nint_fac, "eccentricity":eccentricity}

imax = 16
n_per_i = 8
num = n_per_i * imax + 1

all_args = {
    "nint_fac" : [math.floor(2**i) for i in np.linspace(0, imax, num = num, endpoint=True)],
    "eccentricity" : [0., 0.01, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999],
}

plot_intent = {
    "nint_fac"   : "points"       ,
    "eccentricity"       : "curve_color"  ,
    pyquickbench.out_ax_name : "subplot_grid_y"        ,
}

res = pyquickbench.run_benchmark(
    all_args                                    ,
    [kepler_error]                              ,
    setup = setup                               ,
    mode = "vector_output"                      ,
    filename = bench_filename                   ,
    ForceBenchmark = True                       ,
)


pyquickbench.plot_benchmark(
    res                             ,
    all_args                        ,
    [kepler_error]                  ,
    plot_intent = plot_intent       ,
    show = True                     ,
    # transform = 'pol_cvgence_order' ,
)