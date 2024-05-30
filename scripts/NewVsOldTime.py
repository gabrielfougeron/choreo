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
import mkl_fft

import choreo 

DP_Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.txt")
choreo.find_new.Load_wisdom_file(DP_Wisdom_file)

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

    
def params_to_action_grad_new(NBS, ActionSyst, params_buf_new, params_buf_old):
    NBS.params_to_action_grad(params_buf_new)

def params_to_action_grad_old(NBS, ActionSyst, params_buf_new, params_buf_old):
    ActionSyst.Compute_action(params_buf_old)
    
    
all_funs = {
    'New' : params_to_action_grad_new ,
    'Old' : params_to_action_grad_old ,
}

# mode = 'vector_output'  
mode = 'timings'

def setup(test_name, fft_backend, nint_fac):
    
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
    
    if (inter_pow == -1.) and (inter_pm == 1) :
        inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
        # inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "elastic_pot")
    else:
        inter_law = choreo.numba_funs_new.pow_inter_law(inter_pow/2, inter_pm)
        
    NBS = choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_law)

    # NBS.fftw_planner_effort = 'FFTW_ESTIMATE'
    # NBS.fftw_planner_effort = 'FFTW_MEASURE'
    # NBS.fftw_planner_effort = 'FFTW_PATIENT'
    NBS.fftw_planner_effort = 'FFTW_EXHAUSTIVE'
    
    # NBS.fftw_wisdom_only = False
    NBS.fftw_wisdom_only = True
    
    NBS.fftw_nthreads = 1
    
    NBS.fft_backend = fft_backend
    
    NBS.nint_fac = nint_fac
    
    Sym_list_old = choreo.helper.Make_ChoreoSymList_From_ActionSymList(Sym_list, nbody)
    
    nint_init = NBS.nint
    n_reconverge_it_max = 0

    ActionSyst = choreo.funs.setup_changevar(
        geodim                  ,
        nbody                   ,
        nint_init               ,
        mass                    ,
        n_reconverge_it_max     ,
        Sym_list = Sym_list_old ,
        MomCons = False         ,
        n_grad_change = 1.      ,
        CrashOnIdentity = True  ,
        # ForceMatrixChangevar = True  ,
    )

    # GradHessBackend = "Cython"
    GradHessBackend = "Numba"

    ActionSyst.SetBackend(parallel=False, TwoD=True, GradHessBackend=GradHessBackend)

    ActionSyst.current_cvg_lvl = 0
    
    assert NBS.nint == ActionSyst.nint
    
    # print(NBS.nparams, ActionSyst.nparams)
    # assert NBS.nparams == ActionSyst.nparams
    
    params_buf_new = np.random.random((NBS.nparams))
    params_buf_old = np.random.random((ActionSyst.nparams))
    
    if fft_backend == "scipy":
        
        ActionSyst.rfft = scipy.fft.rfft
        ActionSyst.irfft = scipy.fft.irfft
        
    elif fft_backend == "mkl":
                
        ActionSyst.rfft = mkl_fft._numpy_fft.rfft
        ActionSyst.irfft = mkl_fft._numpy_fft.irfft
        

    
    return {"NBS":NBS, "ActionSyst":ActionSyst, "params_buf_new":params_buf_new, "params_buf_old":params_buf_old}
        

        
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
    # '4D',
    # '3C',
    # '4C',
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
    '3C7k2',
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
    # '20B',
]

min_exp = 0
max_exp = 15

MonotonicAxes = ["nint_fac"]

all_args = {
    "test_name" : all_tests,
    # "fft_backend" : ['scipy', 'mkl', 'fftw'],
    # "fft_backend" : ['scipy', 'mkl'],
    "fft_backend" : ['mkl'],
    # "fft_backend" : ['scipy'],
    # "fft_backend" : ['fftw'],
    "nint_fac" : [2**i for i in range(min_exp,max_exp)] 
}

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

basename = 'NewVsOld_timing'
filename = os.path.join(timings_folder,basename+'.npz')

n_repeat = 1

all_timings = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup           ,
    filename = filename     ,
    # StopOnExcept = True     ,
    ShowProgress = True     ,
    mode = mode  ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    # time_per_test=0.2,
    ForceBenchmark = True,
    # PreventBenchmark = False,
    # ForceBenchmark = False,
    # PreventBenchmark = True,
)

plot_intent = {
    # "test_name" : 'subplot_grid_y'                  ,
    "test_name" : 'curve_linestyle'                  ,
    # "fft_backend" : 'curve_pointstyle'                  ,
    # "fft_backend" : 'curve_color'                  ,
    "fft_backend" : 'subplot_grid_y'                  ,
    "nint_fac" : 'points'                           ,
    pyquickbench.repeat_ax_name :  'reduction_min'  ,
    # pyquickbench.repeat_ax_name :  'reduction_avg'  ,
    # pyquickbench.out_ax_name :  'curve_color'  ,
    # pyquickbench.out_ax_name :  'reduction_sum'  ,
    # pyquickbench.out_ax_name :  'single_value'  ,
}

single_values_val = {
    # pyquickbench.out_ax_name :  'params_to_ifft'  ,
    pyquickbench.out_ax_name :  'ifft_to_params'  ,
}

relative_to_val_list = [
    None    ,
    {pyquickbench.fun_ax_name : 'New'},
    # {"fft_backend" : 'scipy'},
    # {"test_name" : '3C'},
]

# plot_ylim = [1e-6, 8e-3]
# plot_ylim = [1e-7, 3e-3]
plot_ylim = None

for relative_to_val in relative_to_val_list:

    pyquickbench.plot_benchmark(
        all_timings                             ,
        all_args                                ,
        all_funs                                ,
        setup = setup                           ,
        plot_intent = plot_intent               ,
        mode = mode                             ,
        show = True                             ,
        relative_to_val = relative_to_val       ,
        single_values_val = single_values_val   ,
        # transform = "pol_growth_order"          ,
        plot_ylim = plot_ylim                   ,
    )

