import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)
# 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

import pyquickbench
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy

import mkl_fft
scipy.fft.set_global_backend(
    backend = mkl_fft._scipy_fft_backend   ,
    only = True
)


import choreo 


if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

def params_to_pot_nrg_grad(NBS, params_buf):
    NBS.params_to_pot_nrg_grad(params_buf)
    

all_funs = [
    params_to_pot_nrg_grad  ,
]

def setup(test_name, nint_fac):
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', test_name)
    params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
    
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "charge", "Sym_list"])
    
    geodim = all_kwargs["geodim"]
    nbody = all_kwargs["nbody"]
    mass = all_kwargs["mass"]
    charge = all_kwargs["charge"]
    Sym_list = all_kwargs["Sym_list"]
    
    inter_law_cy = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
    
    NBS = choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_law_cy)
    NBS.nint_fac = nint_fac
        
    params_buf = np.random.random((NBS.nparams))

    return {"NBS":NBS, "params_buf":params_buf}
        

        
all_tests = [
    # '3q',
    # '3q3q',
    # '3q3qD',
    # '2q2q',
    # '4q4q',
    # '4q4qD',
    # '4q4qD3k',
    # '1q2q',
    # '5q5q',
    # '6q6q',
    # '2C3C',
    # '2D3D',   
    # '2C3C5k',
    # '2D3D5k',
    # '2D1',
    # '4C5k',
    # '4D3k',
    # '4C',
    # '4D',
    '3C',
    # '3D',
    # '3D1',
    # '3C2k',
    # '3D2k',
    # '3C4k',
    # '3D4k',
    # '3C5k',
    # '3D5k',
    # '3D101k',
    # 'test_3D5k',
    # '3C7k2',
    # '3D7k2',
    # '6C',
    # '6D',
    # '6Ck5',
    # '6Dk5',
    # '5Dq',
    # '2C3C5C',
    # '3C_3dim',
    # '2D1_3dim',
    # '3C',
    # '3C2k',
    # '4C5k',
    # "3C11k",
    # "3C17k",
    # "3C23k",
    # "3C29k",
    # "3C37k",
    # '3C101k',
]

min_exp = 5
max_exp = 10

n_repeat = 1000

MonotonicAxes = ["nint_fac"]

all_args = {
    "test_name" : all_tests,
    "nint_fac" : [2**i for i in range(min_exp,max_exp)] 
}


timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

basename = 'NewSymDev_timing_1'
filename = os.path.join(timings_folder,basename+'.npz')

all_timings_1 = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup           ,
    filename = filename     ,
    ShowProgress = True     ,
    time_per_test = 2.     ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    ForceBenchmark = True,
)


basename = 'NewSymDev_timing_2'
filename = os.path.join(timings_folder,basename+'.npz')

all_timings_2 = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup           ,
    filename = filename     ,
    ShowProgress = True     ,
    time_per_test = 2.     ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    ForceBenchmark = True,
)

# print(all_timings_1.shape)

all_timings = np.concatenate((all_timings_1, all_timings_2), axis=2)


# print(all_timings.shape)

plot_intent = {
    "test_name" : 'subplot_grid_y'                  ,
    "nint_fac" : 'points'                           ,
    pyquickbench.fun_ax_name :  'curve_color'       ,
    pyquickbench.repeat_ax_name :  'reduction_avg'  ,
    # pyquickbench.repeat_ax_name :  'reduction_min'  ,
    # pyquickbench.repeat_ax_name :  'same'  ,
}

relative_to_val_list = [
    None    ,
    {pyquickbench.fun_ax_name : 'fftpack'},
]

for relative_to_val in relative_to_val_list:

    pyquickbench.plot_benchmark(
        all_timings                             ,
        all_args                                ,
        all_fun_names = ["fftpack", "mkl"]                      ,
        plot_intent = plot_intent               ,
        show = True                             ,
        relative_to_val = relative_to_val       ,
            # alpha = 1./255                  ,
    )

