import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

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

import choreo 

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

n_test = 1
    
def params_to_action_grad_scipy(NBS, params_buf):
    
    NBS.fft_backend = 'scipy'

    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        names_reduction = "min" ,
    )
    
    for i in range(n_test):
        NBS.TT_params_to_action_grad(params_buf, TT)
    
    return TT

def params_to_action_grad_mkl(NBS, params_buf):
    
    NBS.fft_backend = 'mkl'

    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        names_reduction = "min" ,
    )
    
    for i in range(n_test):
        NBS.TT_params_to_action_grad(params_buf, TT)
    
    return TT
    
all_funs = [
    params_to_action_grad_scipy ,
    params_to_action_grad_mkl   ,
]

def setup(test_name, nint_fac):
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', test_name)
    params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
    
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "charge", "inter_pow", "inter_pm", "Sym_list"])
    
    geodim = all_kwargs["geodim"]
    nbody = all_kwargs["nbody"]
    mass = all_kwargs["mass"]
    charge = all_kwargs["charge"]
    Sym_list = all_kwargs["Sym_list"]
    
    inter_pow = all_kwargs["inter_pow"]
    inter_pm = all_kwargs["inter_pm"]
    
    inter_pm = -1
    
    if (inter_pow == -1.) and (inter_pm == 1) :
        inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
    else:
        inter_law = choreo.numba_funs_new.pow_inter_law(inter_pow/2, inter_pm)

    NBS = choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_law)
    
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
    # '20B',
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
    # '3C2k',
    # '4C5k',
    # "3C11k",
    # "3C17k",
    # "3C23k",
    # "3C29k",
    # "3C37k",
    # '3C101k',
]

min_exp = 0
max_exp = 23

n_repeat = 1

MonotonicAxes = ["nint_fac"]

all_args = {
    "test_name" : all_tests,
    "nint_fac" : [2**i for i in range(min_exp,max_exp)] 
}


timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

basename = 'NewSymDev_timing'
filename = os.path.join(timings_folder,basename+'.npz')

all_timings = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup           ,
    filename = filename     ,
    # StopOnExcept = True     ,
    ShowProgress = True     ,
    mode = 'vector_output'  ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    # ForceBenchmark = True,
)

plot_intent = {
    "test_name" : 'subplot_grid_y'                  ,
    # "test_name" : 'curve_linestyle'                  ,
    pyquickbench.fun_ax_name : 'curve_linestyle'                  ,
    "nint_fac" : 'points'                           ,
    pyquickbench.repeat_ax_name :  'reduction_min'  ,
    pyquickbench.out_ax_name :  'curve_color'  ,
    # pyquickbench.out_ax_name :  'reduction_sum'  ,
    # pyquickbench.out_ax_name :  'single_value'  ,
}

single_values_val = {
    pyquickbench.out_ax_name :  'params_to_ifft'  ,
    # pyquickbench.out_ax_name :  'ifft_to_params'  ,
}

relative_to_val_list = [
    None    ,
    # {pyquickbench.out_ax_name : 'params_to_ifft'},
    {pyquickbench.fun_ax_name : 'params_to_action_grad_scipy'},
]

for relative_to_val in relative_to_val_list:

    pyquickbench.plot_benchmark(
        all_timings                             ,
        all_args                                ,
        all_funs                                ,
        setup = setup                           ,
        plot_intent = plot_intent               ,
        mode = 'vector_output'                  ,
        show = True                             ,
        relative_to_val = relative_to_val       ,
        single_values_val = single_values_val   ,
    )

