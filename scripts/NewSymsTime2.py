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

n_repeat = 1


def TransformSegment_old(Sym, segmpos_in, segmpos_out):
    Sym.TransformSegment_old(segmpos_in, segmpos_out)
    
def TransformSegment(Sym, segmpos_in, segmpos_out):
    Sym.TransformSegment(segmpos_in, segmpos_out)
    
    
all_funs = [
    TransformSegment_old ,
    TransformSegment ,
]

# mode = 'vector_output'  
mode = 'timings'

def setup(segm_store, geodim, TimeRev):
    
    Sym = choreo.ActionSym(
        BodyPerm  = np.array(range(3), dtype = np.intp)     ,
        SpaceRot  = np.random.random((geodim, geodim))      ,
        TimeRev   = TimeRev                                 ,
        TimeShiftNum = 0                                    ,
        TimeShiftDen = 1                                    ,
    )

    segmpos_in = np.random.random((segm_store, geodim))
    segmpos_out = np.empty((segm_store, geodim))
    
    return {"Sym":Sym, "segmpos_in":segmpos_in, "segmpos_out": segmpos_out}
        
min_exp = 0
max_exp = 20
# max_exp = 5

MonotonicAxes = ["segm_store"]

all_args = {
    "segm_store" : [2**i for i in range(min_exp,max_exp)] ,
    "geodim" : [1,2,3,4] ,
    "TimeRev" : [1,-1] ,
}

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

basename = 'NewSymDev_timing2'
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
    ForceBenchmark = True,
    # PreventBenchmark = False,
    # ForceBenchmark = False,
    # PreventBenchmark = True,
)

plot_intent = {
    "segm_store" : 'points'                           ,
    "geodim" : 'subplot_grid_y'                           ,
    "TimeRev" : 'curve_linestyle'                           ,
    pyquickbench.fun_ax_name :  'curve_color'  ,
    pyquickbench.repeat_ax_name :  'reduction_min'  ,
    # pyquickbench.repeat_ax_name :  'reduction_avg'  ,
    # pyquickbench.out_ax_name :  'curve_color'  ,
    # pyquickbench.out_ax_name :  'reduction_sum'  ,
    # pyquickbench.out_ax_name :  'single_value'  ,
}

relative_to_val_list = [
    None    ,
    # {
    #     "fft_backend" : 'scipy',
    #     "ForceGeneralSym" : True,
    # },
    # {"fft_backend" : 'scipy'},
    # {"fft_backend" : 'mkl'},
    # {"test_name" : '3C'},
]

# plot_ylim = [7e-6, 1.5e-1]
# plot_ylim = [3e-7, 8e-3]
# plot_ylim = [0., 0.5]
# plot_ylim = [0.1, 1.1]
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
        # transform = "relative_curve_fraction"   ,
        plot_ylim = plot_ylim                   ,
    )
