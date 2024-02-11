import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

import choreo 
import pyquickbench
import json
import matplotlib.pyplot as plt
import mkl_fft
import scipy
scipy.fft.set_global_backend(
    backend = mkl_fft._scipy_fft_backend   ,
    only = True
)


if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

all_funs = {
    # 'all_pos_full'  : choreo.funs_new.params_to_all_pos,
    # 'all_pos_full_mod'  : choreo.funs_new.params_to_all_pos_mod,
    
    
    'all_pos_slice' : choreo.funs_new.params_to_all_pos_slice,
    'all_pos_slice_mod'  : choreo.funs_new.params_to_all_pos_slice_mod,
    'all_pos_slice_mod2'  : choreo.funs_new.params_to_all_pos_slice_mod2,
    'all_pos_slice_mod3'  : choreo.funs_new.params_to_all_pos_slice_mod3,
}

def setup(test_name, nint_fac):
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'NewSym_data', 'tests')
    params_filename = os.path.join(__PROJECT_ROOT__, 'NewSym_data', test_name, "choreo_config.json")
    
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)
        
    all_kwargs_speed = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, callback=choreo.funs_new.Prepare_data_for_speed_comparison)
    
    all_kwargs_speed['nint_fac'] = nint_fac
    
    all_kwargs = choreo.funs_new.Prepare_data_for_speed_comparison(**all_kwargs_speed)
    
    return all_kwargs
        

        
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
    '2C3C',
    # '2D3D',   
    # '2C3C5k',
    # '2D3D5k',
    # '2D1',
    # '4C5k',
    # '4D3k',
    # '4C',
    # '4D',
    # '3C',
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

min_exp = 7
max_exp = 12

n_repeat = 10

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
    setup = setup       ,
    filename = filename     ,
    StopOnExcept = True     ,
    ShowProgress = True     ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    ForceBenchmark = True
)

plot_intent = {
    "test_name" : 'subplot_grid_y'                  ,
    "nint_fac" : 'points'                           ,
    pyquickbench.fun_ax_name :  'curve_color'       ,
    pyquickbench.repeat_ax_name :  'reduction_min'  ,
}

relative_to_val_list = [
    None    ,
    {pyquickbench.fun_ax_name :  'all_pos_slice'},
]

for relative_to_val in relative_to_val_list:

    pyquickbench.plot_benchmark(
        all_timings                             ,
        all_args                                ,
        all_funs                                ,
        plot_intent = plot_intent               ,
        show = True                             ,
        relative_to_val = relative_to_val       ,
    )

