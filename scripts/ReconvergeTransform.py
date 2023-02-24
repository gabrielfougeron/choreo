import os

import concurrent.futures
import multiprocessing
import json
import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions
import scipy.integrate
import scipy.special

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime

One_sec = 1e9

def main():



    # input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/04 - Montaldi-Steckles-Gries')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/9')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/Keep/')
    input_folder = os.path.join(__PROJECT_ROOT__,'Reconverged_sols')
    
#     ''' Include all files in tree '''
#     input_names_list = []
#     for root, dirnames, filenames in os.walk(input_folder):
# 
#         for filename in filenames:
#             file_path = os.path.join(root, filename)
#             file_root, file_ext = os.path.splitext(os.path.basename(file_path))
# 
#             if (file_ext == '.json' ):
# 
#                 file_path = os.path.join(root, file_root)
#                 the_name = file_path[len(input_folder):]
#                 input_names_list.append(the_name)
# 
# # # 
    ''' Include all files in folder '''
    input_names_list = []
    for file_path in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.json' ):
            # 
            # if int(file_root) > 8:
            #     input_names_list.append(file_root)

            input_names_list.append(file_root)

    # input_names_list = ['01 - Figure eight']
    # input_names_list = ['04 - 5 pointed star']
    # input_names_list = ['14 - Small mass gap']


    store_folder = os.path.join(__PROJECT_ROOT__,'Reconverged_sols')
    # store_folder = input_folder

    Exec_Mul_Proc = True
    # Exec_Mul_Proc = False

    if Exec_Mul_Proc:

        # n = 1
        # n = 4
        # n = multiprocessing.cpu_count()
        n = multiprocessing.cpu_count()//2
        
        print(f"Executing with {n} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
            
            res = []
            
            for the_name in input_names_list:

                all_kwargs = choreo.Pick_Named_Args_From_Dict(ExecName,dict(globals(),**locals()))
                res.append(executor.submit(ExecName,**all_kwargs))
                time.sleep(0.01)

    else:
            
        for the_name in input_names_list:

            all_kwargs = choreo.Pick_Named_Args_From_Dict(ExecName,dict(globals(),**locals()))
            ExecName(the_name, input_folder, store_folder)


def ExecName(the_name, input_folder, store_folder):

    print('--------------------------------------------')
    print('')
    print(the_name)
    print('')
    print('--------------------------------------------')
    print('')

    file_basename = the_name
    
    Info_filename = os.path.join(input_folder,the_name + '.json')

    with open(Info_filename,'r') as jsonFile:
        Info_dict = json.load(jsonFile)


    input_filename = os.path.join(input_folder,the_name + '.npy')

    bare_name = the_name.split('/')[-1]

    all_pos = np.load(input_filename)
    nint_init = Info_dict["n_int"]
    ncoeff_init = nint_init // 2 +1

    c_coeffs = choreo.the_rfft(all_pos,axis=2,norm="forward")
    all_coeffs = np.zeros((Info_dict["nloop"],choreo.ndim,ncoeff_init,2),dtype=np.float64)
    all_coeffs[:,:,:,0] = c_coeffs.real
    all_coeffs[:,:,:,1] = c_coeffs.imag


    # theta = 2*np.pi * 0.
    # SpaceRevscal = 1.
    # SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    # TimeRev = 1.
    # TimeShiftNum = 0
    # TimeShiftDen = 1


    theta = 2*np.pi * 0/2
    SpaceRevscal = 1.
    SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    TimeRev = 1.
    TimeShiftNum = 0
    TimeShiftDen = 2



    all_coeffs_init = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)
    Transform_Sym = choreo.ChoreoSym(SpaceRot=SpaceRot, TimeRev=TimeRev, TimeShift = fractions.Fraction(numerator=TimeShiftNum,denominator=TimeShiftDen))

    Transform_Sym = None

    ReconvergeSol = True


    nbody = Info_dict['nbody']
    mass = np.array(Info_dict['mass']).astype(np.float64)
    Sym_list = choreo.Make_SymList_From_InfoDict(Info_dict,Transform_Sym)


    MomConsImposed = True
    # MomConsImposed = False
# 
#     rot_angle = 0
#     s = -1
# 
#     Sym_list.append(choreo.ChoreoSym(
#         LoopTarget=0,
#         LoopSource=0,
#         SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
#         TimeRev=-1,
#         TimeShift=fractions.Fraction(numerator=0,denominator=1)
#     ))


    Save_All_Pos = True
    # Save_All_Pos = False

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_All_Coeffs_No_Sym = True
    Save_All_Coeffs_No_Sym = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    Save_img = True
    # Save_img = False
# 
    # Save_thumb = True
    Save_thumb = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches
    thumb_size = (2,2) # Image size in inches
    
    color = "body"
    # color = "loop"
    # color = "velocity"
    # color = "all"

    # Save_anim = True
    Save_anim = False


    vid_size = (8,8) # Image size in inches
    # nint_plot_anim = 2*2*2*3*3
    nint_plot_anim = 2*2*2*3*3*5
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint

    min_n_steps_ode = 1*nint_plot_anim

    nperiod_anim = 1.

    Plot_trace_anim = True
    # Plot_trace_anim = False
# 
    # InvestigateStability = True
    InvestigateStability = False

    # Save_Perturbed = True
    Save_Perturbed = False


    Use_exact_Jacobian = True
    # Use_exact_Jacobian = False

    # Look_for_duplicates = True
    Look_for_duplicates = False

    Check_Escape = True
    # Check_Escape = False

    # Penalize_Escape = True
    Penalize_Escape = False

    save_first_init = False
    # save_first_init = True

    save_all_inits = False
    # save_all_inits = True

    # max_norm_on_entry = 1e-6
    max_norm_on_entry = 1e6

    mul_coarse_to_fine = 3

    n_grad_change = 1.
    # n_grad_change = 0.

    coeff_ampl_min  = 1e-16
    coeff_ampl_o    = 1e-16
    k_infl          = 2
    k_max           = 3


    duplicate_eps = 1e-8
    freq_erase_dict = 100
    hash_dict = {}

    n_opt = 0
    n_opt_max = 1
    n_find_max = 1



    n_reconverge_it_max = 1


    Newt_err_norm_max = 1e-13
    Newt_err_norm_max_save = Info_dict['Newton_Error']

    krylov_method = 'lgmres'
    # krylov_method = 'gmres'
    # krylov_method = 'bicgstab'
    # krylov_method = 'cgs'
    # krylov_method = 'minres'
    # krylov_method = 'tfqmr'


    # line_search = 'armijo'
    # line_search = 'wolfe'
    line_search = 'none'

    # disp_scipy_opt = False
    disp_scipy_opt = True

    # linesearch_smin = 0.1
    linesearch_smin = 1
    
    gradtol_list =          [1e-1   ,1e-3   ,1e-5   ,1e-7   ,1e-9   ,1e-11  ,1e-13  ,1e-15  ]
    inner_maxiter_list =    [30     ,30     ,50     ,60     ,70     ,80     ,100    ,100    ]
    maxiter_list =          [100    ,1000   ,1000   ,1000   ,500    ,500    ,300    ,100    ]
    outer_k_list =          [5      ,5      ,5      ,5      ,5      ,7      ,7      ,7      ]
    store_outer_Av_list =   [False  ,False  ,False  ,False  ,False  ,True   ,True   ,True   ]
    # 
    # gradtol_list =          [1e-1   ,1e-3  ,1e-5  ]
    # inner_maxiter_list =    [30     ,30    ,50    ]
    # maxiter_list =          [100    ,1000  ,1000  ]
    # outer_k_list =          [5      ,5     ,5     ]
    # store_outer_Av_list =   [False  ,False ,False ]

    
    n_optim_param = len(gradtol_list)
    
    gradtol_max = 100*gradtol_list[n_optim_param-1]
    # foundsol_tol = 1000*gradtol_list[0]
    foundsol_tol = 1e10
    plot_extend = 0.

    AddNumberToOutputName = False

    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(**locals()))
    choreo.Find_Choreo(**all_kwargs)

#     filename_output = store_folder+'/'+file_basename
#     filename = filename_output+'.npy'
#     all_pos_post_reconverge = np.load(filename)
# 
#     print(np.linalg.norm(all_pos_post_reconverge - all_pos))
# 
# 
# 











if __name__ == "__main__":
    main()    
