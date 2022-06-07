import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
import shutil
import random
import time
import math as m
import numpy as np
import sys
import fractions

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

def main():
    

    # if (the_i != 0):
        
        # preprint_msg = str(the_i).zfill(2)+' : '

        # def print(*args, **kwargs):
            # """My custom print() function."""
            # builtins.print(preprint_msg,end='')
            # return builtins.print(*args, **kwargs)
            
    file_basename = ''
    
    LookForTarget = False
    
    # nbpl=[1,2,3,4,5,6]
    nbpl=[3]
    # nbpl=[5]
    # nbpl=[2,3,5]



    the_lcm = m.lcm(*nbpl)

    SymName = 'C'
# 
#     SymType = []
# 
#     SymType.append({
#         'name'  : 'D',
#         'n'     : 1,
#         'k'     : 2,
#         'l'     : 1 ,
#         'p'     : 1 ,
#         'q'     : 5,
#     })

    SymType = {
        'name'  : 'D',
        'n'     : 1,
        'k'     : 1,
        'l'     : 1 ,
        'p'     : 0 ,
        'q'     : 1,
    }

    # Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)
    Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymType=SymType)

    mass = np.ones((nbody),dtype=np.float64)

    # rot_angle =  0
    # s = -1
    #     
    # Sym_list.append(
    #     choreo.ChoreoSym(
    #         LoopTarget=0,
    #         LoopSource=1,
    #         SpaceRot= np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
    #         TimeRev=-1,
    #         TimeShift=fractions.Fraction(numerator=0,denominator=1)
    #         )
    # )        
    # Sym_list.append(
    #     choreo.ChoreoSym(
    #         LoopTarget=2,
    #         LoopSource=2,
    #         SpaceRot= np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
    #         TimeRev=-1,
    #         TimeShift=fractions.Fraction(numerator=0,denominator=1)
    #         )
    # )
    # Sym_list.append(
    #     choreo.ChoreoSym(
    #         LoopTarget=3,
    #         LoopSource=3,
    #         SpaceRot= np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
    #         TimeRev=-1,
    #         TimeShift=fractions.Fraction(numerator=0,denominator=1)
    #         )
    # )


    MomConsImposed = True
    # MomConsImposed = False

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    Use_exact_Jacobian = True
    # Use_exact_Jacobian = False

    Look_for_duplicates = True
    # Look_for_duplicates = False

    Check_Escape = True
    # Check_Escape = False

    # Penalize_Escape = True
    Penalize_Escape = False

    save_init = False
    # save_init = True

    Save_img = True
    # Save_img = False

    # Save_thumb = True
    Save_thumb = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches
    thumb_size = (2,2) # Image size in inches


    
    color = "body"
    # color = "loop"
    # color = "velocity"
    # color = "all"

    Save_anim = True
    # Save_anim = False

    vid_size = (8,8) # Image size in inches
    nint_plot_anim = 2*2*2*3*3*5*2
    # nperiod_anim = 1./nbody
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint

    try:
        the_lcm
    except NameError:
        period_div = 1.
    else:
        period_div = the_lcm

    nperiod_anim = 1.
    # nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # Plot_trace_anim = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    n_reconverge_it_max = 4
    # n_reconverge_it_max = 1

    # ncoeff_init = 102
    # ncoeff_init = 800
    # ncoeff_init = 201   
    ncoeff_init = 300   
    # ncoeff_init = 600
    # ncoeff_init = 900
    # ncoeff_init = 1800
    # ncoeff_init = 2400
    # ncoeff_init = 1206
    # ncoeff_init = 9*8*7*5
    # ncoeff_init = the_lcm
    
    # print(the_lcm)

    disp_scipy_opt = False
    # disp_scipy_opt = True
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = 1e-10
    # Newt_err_norm_max_save = Newt_err_norm_max*1000
    Newt_err_norm_max_save = 1e-8

    duplicate_eps = 1e-8

    krylov_method = 'lgmres'
    # krylov_method = 'gmres'
    # krylov_method = 'bicgstab'
    # krylov_method = 'cgs'
    # krylov_method = 'minres'

    # line_search = 'armijo'
    line_search = 'wolfe'
    
    gradtol_list =          [1e-1   ,1e-3   ,1e-5   ,1e-7   ,1e-9   ,1e-11  ,1e-13  ,1e-15  ]
    inner_maxiter_list =    [30     ,30     ,50     ,60     ,70     ,80     ,100    ,100    ]
    maxiter_list =          [100    ,1000   ,1000   ,1000   ,500    ,500    ,300    ,100    ]
    outer_k_list =          [5      ,5      ,5      ,5      ,5      ,7      ,7      ,7      ]
    store_outer_Av_list =   [False  ,False  ,False  ,False  ,False  ,True   ,True   ,True   ]
    
    n_optim_param = len(gradtol_list)
    
    gradtol_max = 100*gradtol_list[n_optim_param-1]
    # foundsol_tol = 1000*gradtol_list[0]
    foundsol_tol = 1e10

    escape_fac = 1e0
    # escape_fac = 1e-1
    # escape_fac = 1e-2
    # escape_fac = 1e-3
    # escape_fac = 1e-4
    # escape_fac = 1e-5
    # escape_fac = 0
    escape_min_dist = 1
    escape_pow = 2.0
    # escape_pow = 2.5
    # escape_pow = 1.5
    # escape_pow = 0.5

    n_grad_change = 1.
    # n_grad_change = 1.5

    coeff_ampl_o=1e-1
    # coeff_ampl_o=1e0
    k_infl=3
    # k_max=600
    # k_max=200
    k_max=600
    coeff_ampl_min=1e-16

    freq_erase_dict = 100
    hash_dict = {}

    n_opt = 0
    # n_opt_max = 1
    # n_opt_max = 5
    n_opt_max = 1e10
    # n_opt_max = 0
    
    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.GenSymExample,dict(globals(),**locals()))
    
    choreo.GenSymExample(**all_kwargs)




if __name__ == "__main__":
    main()

