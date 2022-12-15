
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
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)
    
import choreo 

def main(the_i=0):

    file_basename = ''
    
    np.random.seed(int(time.time()*10000) % 5000)

    LookForTarget = True
    

    # slow_base_filename = './choreo_GUI/choreo-gallery/02 - Helpers/01 - Circles/01'
    # slow_base_filename = './choreo_GUI/choreo-gallery/02 - Helpers/01 - Circles/02'
    slow_base_filename = './choreo_GUI/choreo-gallery/01 - Classic gallery/01 - Figure eight'
    # slow_base_filename = './choreo_GUI/choreo-gallery/01 - Classic gallery/06 - 5-chain'

    # fast_base_filename_list = ['./choreo_GUI/choreo-gallery/02 - Helpers/01 - Circles/02'   ] 

    fast_base_filename_list = [
        # './choreo_GUI/choreo-gallery/02 - Helpers/01 - Circles/01',
        # './choreo_GUI/choreo-gallery/02 - Helpers/01 - Circles/02',
        # './choreo_GUI/choreo-gallery/02 - Helpers/01 - Circles/03',
        './choreo_GUI/choreo-gallery/01 - Classic gallery/01 - Figure eight'
        ] 

    



    Info_slow_filename = slow_base_filename + '.json'

    with open(Info_slow_filename,'r') as jsonFile:
        Info_dict_slow = json.load(jsonFile)

    input_slow_filename = slow_base_filename + '.npy'

    all_pos_slow = np.load(input_slow_filename)
    nint_slow = Info_dict_slow["n_int"]
    ncoeffs_slow = Info_dict_slow["n_Fourier"]
    nloop_slow = Info_dict_slow["nloop"]

    c_coeffs_slow = choreo.the_rfft(all_pos_slow,n=nint_slow,axis=2,norm="forward")
    all_coeffs_slow_load = np.zeros((nloop_slow,choreo.ndim,ncoeffs_slow,2),dtype=np.float64)
    all_coeffs_slow_load[:,:,:,0] = c_coeffs_slow[:,:,0:ncoeffs_slow].real
    all_coeffs_slow_load[:,:,:,1] = c_coeffs_slow[:,:,0:ncoeffs_slow].imag

    all_coeffs_fast_list = []
    Info_dict_fast_list = []
    for fast_base_filename in fast_base_filename_list :

        Info_fast_filename = fast_base_filename + '.json'

        with open(Info_fast_filename,'r') as jsonFile:
            Info_dict_fast = json.load(jsonFile)

        input_fast_filename = fast_base_filename + '.npy'

        all_pos_fast = np.load(input_fast_filename)

        nint_fast = Info_dict_fast["n_int"]

        c_coeffs_fast = choreo.the_rfft(all_pos_fast,n=nint_fast,axis=2,norm="forward")
        all_coeffs_fast = np.zeros((Info_dict_fast["nloop"],choreo.ndim,Info_dict_fast["n_Fourier"],2),dtype=np.float64)
        all_coeffs_fast[:,:,:,0] = c_coeffs_fast[:,:,0:Info_dict_fast["n_Fourier"]].real
        all_coeffs_fast[:,:,:,1] = c_coeffs_fast[:,:,0:Info_dict_fast["n_Fourier"]].imag

        all_coeffs_fast_list.append(all_coeffs_fast)
        Info_dict_fast_list.append(Info_dict_fast)






    nfl = len(fast_base_filename_list)

    mass_mul = [1]
    nTf = [37]
    nbs = [3]
    nbf = [3]

    epsmul = 0.

    # mass_mul = [1,1,1,1]
    # mass_mul = [1.,1.+epsmul,1.+2*epsmul,1.+3*epsmul]
    # nTf = [1,1,1,1]
    # nbs = [1,1,1,1]
    # nbf = [1,1,1,1]

    # mass_mul = [1,1]
    # mass_mul = [3,2]
    # nTf = [37,37]
    # nbs = [1,1]
    # nbf = [2,3]

    mul_loops_ini = True
    # mul_loops_ini = False
    # mul_loops_ini = (np.random.random() > 1./2.)
    
    mul_loops = [mul_loops_ini for _ in range(nfl)]

    Remove_Choreo_Sym = mul_loops
    # Remove_Choreo_Sym = [False,False]
    # Remove_Choreo_Sym = [False,False]

    Rotate_fast_with_slow = True
    # Rotate_fast_with_slow = False
    # Rotate_fast_with_slow = (np.random.random() > 1./2.)

    Optimize_Init = True
    # Optimize_Init = False
    # Optimize_Init = (np.random.random() > 1./2.)

    Randomize_Fast_Init = True
    # Randomize_Fast_Init = False

    


    all_coeffs_fast_load_list = []

    nbpl=[]
    mass = []
    for i in range(nfl):
        all_coeffs_fast_load = all_coeffs_fast_list[i]

        if Remove_Choreo_Sym[i]:
            
            all_coeffs_fast_load_list_temp = []

            for ib in range(nbf[i]):
                
                tshift = (ib*1.0)/nbf[i]

                all_coeffs_fast_load_list_temp.append(choreo.Transform_Coeffs(np.identity(2), 1, tshift, 1, all_coeffs_fast_load))

                nbpl.append(nbs[i])
                mass.extend([mass_mul[i] for j in range(nbs[i])])

            all_coeffs_fast_load_list.append(np.concatenate(all_coeffs_fast_load_list_temp,axis=0))

        else:
            all_coeffs_fast_load_list.append( all_coeffs_fast_load)
            nbpl.append(nbs[i]*nbf[i])
            mass.extend([mass_mul[i] for j in range(nbs[i]*nbf[i])])

    mass = np.array(mass,dtype=np.float64)

    Sym_list = []
    the_lcm = m.lcm(*nbpl)
    SymName = None
    Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

    # mass = np.ones((nbody))*mass_mul


#     for ibody in range(6):
#     # for ibody in [0,2]:
#     # for ibody in [0]:
# 
#         l_rot = 11
#         k_rot = 13
#         rot_angle = 2* np.pi * l_rot / k_rot
#         s = 1
#         st = 1
# 
#         Sym_list.append(choreo.ChoreoSym(
#                 LoopTarget=ibody,
#                 LoopSource=ibody,
#                 SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
#                 TimeRev=st,
#                 TimeShift=fractions.Fraction(numerator=1,denominator=k_rot)
#             ))
# 

    # MomConsImposed = True
    MomConsImposed = False

    store_folder = './Target_res/'
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
# 
    # save_first_init = False
    save_first_init = True
# 
    save_all_inits = False
    # save_all_inits = True

    Save_img = True
    # Save_img = False

    # Save_thumb = True
    Save_thumb = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches
    thumb_size = (2,2) # Image size in inches

    nint_plot_img = 10000
    
    color = "body"
    # color = "loop"
    # color = "velocity"
    # color = "all"

    Save_anim = True
    # Save_anim = False

    Save_All_Coeffs = False
    Save_All_Pos = True

    vid_size = (8,8) # Image size in inches
    nint_plot_anim = 2*2*2*3*3*5 
    # nperiod_anim = 1./nbody
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint

    try:
        the_lcm
    except NameError:
        period_div = 1.
    else:
        period_div = the_lcm

    nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # Plot_trace_anim = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    n_reconverge_it_max = 3
    # n_reconverge_it_max = 1

    # ncoeff_init = 102
    # ncoeff_init = 800
    # ncoeff_init = 201   
    # ncoeff_init = 300   
    # ncoeff_init = 600
    # ncoeff_init = 900
    ncoeff_init = 1800
    # ncoeff_init = 3600
    # ncoeff_init = 7200
    # ncoeff_init = 1206
    # ncoeff_init = 90

    # disp_scipy_opt = False
    disp_scipy_opt = True
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = 1e-11
    # Newt_err_norm_max_save = Newt_err_norm_max*1000
    Newt_err_norm_max_save = 1e-1

    duplicate_eps = 1e-8

    krylov_method = 'lgmres'
    # krylov_method = 'gmres'
    # krylov_method = 'bicgstab'
    # krylov_method = 'cgs'
    # krylov_method = 'minres'

    # line_search = 'armijo'
    line_search = 'wolfe'
    # line_search = 'none'
 
    # linesearch_smin = 1e-2
    # linesearch_smin = 1.
    linesearch_smin = 0.01
    
    gradtol_list =          [1e-3   ,1e-5   ,1e-7   ,1e-9   ,1e-11  ,1e-13  ,1e-15  ]
    inner_maxiter_list =    [30     ,50     ,60     ,70     ,80     ,100    ,100    ]
    maxiter_list =          [1000   ,1000   ,1000   ,500   ,500     ,300    ,100    ]
    outer_k_list =          [5      ,5      ,5      ,5      ,7      ,7      ,7      ]
    store_outer_Av_list =   [False  ,False  ,False  ,False  ,True   ,True   ,True   ]
    
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
    # n_grad_change = 2.
    # n_grad_change = 0.7

    coeff_ampl_o=1e-16
    k_infl=200
    k_max=800
    coeff_ampl_min=1e-16

    freq_erase_dict = 1000
    hash_dict = {}

    n_opt = 0

    # n_opt_max = 5
    # n_find_max = 1
    
    n_opt_max = 5000
    n_find_max = 100

    mul_coarse_to_fine = 3

    plot_extend = 0.

    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(globals(),**locals()))
    
    choreo.Find_Choreo(**all_kwargs)
# # 
if __name__ == "__main__":
    main(0)
#   
# # 
# if __name__ == "__main__":
# 
#     # n = multiprocessing.cpu_count()
#     n = multiprocessing.cpu_count()//2
#     # n = 4
#     
#     print(f"Executing with {n} workers")
#     
#     
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
#         
#         res = []
#         for i in range(1,n+1):
#             res.append(executor.submit(main,i))
#             time.sleep(0.01)
# 
#  
