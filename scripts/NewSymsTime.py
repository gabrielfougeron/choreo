import os
import sys
import threadpoolctl

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

threadpoolctl.threadpool_limits(limits=1).__enter__()

import pyquickbench
import json
import numpy as np
import matplotlib.pyplot as plt

import choreo 

Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
choreo.find.Load_wisdom_file(Wisdom_file)

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

n_test = 1
n_repeat = 100
    
def params_to_action_grad_TT(NBS, params_buf):

    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        names_reduction = "min" ,
        # names_reduction = "avg" ,
        ignore_names = "start" ,
    )
    
    for i in range(n_test):
        NBS.TT_params_to_action_grad(params_buf, TT)
    
    return TT

    
def params_to_action_grad(NBS, params_buf):
    
    NBS.params_to_action_grad(params_buf)
    
    
all_funs = [
    params_to_action_grad_TT ,
    # params_to_action_grad ,
]

mode = 'vector_output'  
# mode = 'timings'

def setup(test_name, fft_backend, nint_fac, ForceGeneralSym):
    
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
        inter_law_str = "gravity_pot"
        inter_law_param_dict = None
    else:
        inter_law_str = "power_law_pot"
        inter_law_param_dict = {'n': inter_pow, 'alpha': inter_pm}

    NBS = choreo.NBodySyst(
        geodim, nbody, mass, charge, Sym_list,
        inter_law_str = inter_law_str, inter_law_param_dict = inter_law_param_dict,
        ForceGeneralSym = ForceGeneralSym,
    )

    # NBS.fftw_planner_effort = 'FFTW_ESTIMATE'
    # NBS.fftw_planner_effort = 'FFTW_MEASURE'
    # NBS.fftw_planner_effort = 'FFTW_PATIENT'
    NBS.fftw_planner_effort = 'FFTW_EXHAUSTIVE'
    
    # NBS.fftw_wisdom_only = False
    NBS.fftw_wisdom_only = True
    
    NBS.fftw_nthreads = 1
    
    NBS.fft_backend = fft_backend
    
    NBS.nint_fac = nint_fac
        
    params_buf = np.random.random((NBS.nparams))
    
    return {"NBS":NBS, "params_buf":params_buf}
        


all_tests = [
    # 'test'      ,
    '3C'      ,
    # '3D'      ,
    # '3D1'     ,
    # '3C2k'    ,
    # '3D2k'      ,
]

min_exp = 0
# max_exp = 13
max_exp = 18

MonotonicAxes = ["nint_fac"]

all_args = {
    "test_name" : all_tests,
    "fft_backend" : ['scipy', 'mkl', 'fftw', 'ducc'],
    # "fft_backend" : ['scipy', 'mkl'],
    # "fft_backend" : ['scipy'],
    # "fft_backend" : ['mkl'],
    # "fft_backend" : ['fftw'],
    "nint_fac" : [2**i for i in range(min_exp,max_exp)] ,
    "ForceGeneralSym" : [True, False],
    # "ForceGeneralSym" : [False],
}

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

basename = 'NewSymDev_timing'
filename = os.path.join(timings_folder,basename+'.npz')

all_timings = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup           ,
    filename = filename     ,
    mode = mode             ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    time_per_test=0.2       ,
    ShowProgress=True       ,
    # timeout = 1.,
    # ForceBenchmark = True,
    # PreventBenchmark = False,
    # ForceBenchmark = False,
    PreventBenchmark = True,
)

plot_intent = {
    "test_name" : 'subplot_grid_y'                  ,
    # "test_name" : 'curve_linestyle'                  ,
    # "fft_backend" : 'curve_pointstyle'                  ,
    # "fft_backend" : 'curve_color'                  ,
    "fft_backend" : 'curve_linestyle'                  ,
    # "fft_backend" : 'subplot_grid_y'                  ,
    "nint_fac" : 'points'                           ,
    # "ForceGeneralSym" : 'subplot_grid_y'                           ,
    "ForceGeneralSym" : 'curve_color'                           ,
    # pyquickbench.repeat_ax_name :  'reduction_min'  ,
    pyquickbench.repeat_ax_name :  'reduction_avg'  ,
    # pyquickbench.out_ax_name :  'curve_color'  ,
    pyquickbench.out_ax_name :  'reduction_sum'  ,
    # pyquickbench.out_ax_name :  'single_value'  ,
}

single_values_val = {
    # pyquickbench.out_ax_name :  'segm_pos_to_pot_nrg_grad'  ,
    pyquickbench.out_ax_name :  'pos_slice_to_params'  ,
    # pyquickbench.out_ax_name :  'params_to_pos_slice'  ,
}

relative_to_val_list = [
    # None    ,
    {
        "fft_backend" : 'scipy',
        "ForceGeneralSym" : True,
    },
    # {"fft_backend" : 'scipy'},
    # {"fft_backend" : 'mkl'},
    # {"test_name" : '3C'},
]

# plot_ylim = [1e-6, 1e-1]
# plot_ylim = [3e-7, 8e-3]
# plot_ylim = [0., 0.5]
plot_ylim = [0.1, 1.1]
# plot_ylim = None

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
        # transform = "relative_curve_fraction"   ,
        plot_ylim = plot_ylim                   ,
    )
