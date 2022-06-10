
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

def main(the_i=0):
    
    # if (the_i != 0):
    #     
    #     preprint_msg = str(the_i).zfill(2)+' : '
    # 
    #     def print(*args, **kwargs):
    #         """My custom print() function."""
    #         builtins.print(preprint_msg,end='')
    #         return builtins.print(*args, **kwargs)
    # 
    file_basename = ''
    
    np.random.seed(int(time.time()*10000) % 5000)

    LookForTarget = True
    

    # slow_base_filename = './data/1_lone_wolf.npy'
    # slow_base_filename = './data/1_1_short_ellipse.npy'
    # slow_base_filename = './data/1_1_long_ellipse.npy'
    slow_base_filename = './data/1_1_cercle.npy'
    # slow_base_filename = './data/2_cercle.npy'
    # slow_base_filename = './data/3_cercle.npy'
    # slow_base_filename = './data/3_huit.npy'
    # slow_base_filename = './data/3_heart.npy'
    # slow_base_filename = './data/4_trefoil.npy'
    # slow_base_filename = './data/1x4_trefoil.npy'


    # fast_base_filename_list = ['./data/1_lone_wolf.npy'    ] 
    # fast_base_filename_list = ['./data/2_cercle.npy'       ]
    # fast_base_filename_list = ['./data/3_cercle.npy'       ]
    # fast_base_filename_list = ['./data/3_huit.npy'         ]
    # fast_base_filename_list = ['./data/3_heart.npy'        ]
    # fast_base_filename_list = ['./data/3_dbl_heart.npy'    ]
    # fast_base_filename_list = ['./data/4_13_2_2_cercle.npy'] 
    # fast_base_filename_list = ['./data/4_trefoil.npy'] 

    fast_base_filename_list = ['./data/2_cercle.npy','./data/2_cercle.npy'    ] 
    # fast_base_filename_list = ['./data/2_cercle.npy','./data/3_huit.npy'    ] 
    # fast_base_filename_list = ['./data/1_lone_wolf.npy','./data/1_lone_wolf.npy'    ] 
    
    
    # fast_base_filename_list = ['./data/1_lone_wolf.npy','./data/1_lone_wolf.npy'    ,'./data/1_lone_wolf.npy','./data/1_lone_wolf.npy'    ] 
    # fast_base_filename_list = ['./data/1_lone_wolf.npy','./data/2_cercle.npy'    ,'./data/1_lone_wolf.npy','./data/2_cercle.npy'    ] 
    # fast_base_filename_list = ['./data/1_lone_wolf.npy','./data/1_lone_wolf.npy'    ,'./data/1_lone_wolf.npy','./data/3_huit.npy'    ] 

    nfl = len(fast_base_filename_list)

    mass_mul = [1,1]
    nTf = [13,13]
    nbs = [1,1]
    nbf = [2,2]

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

    all_coeffs_slow_load = np.load(slow_base_filename)
    all_coeffs_fast_load_list = []

    nbpl=[]
    mass = []
    for i in range(nfl):
        fast_base_filename = fast_base_filename_list[i]
        all_coeffs_fast_load = np.load(fast_base_filename)

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


    for ibody in [0,1,2,3]:
    # for ibody in [0,2]:
    # for ibody in [0]:

        l_rot = 11
        k_rot = 13
        rot_angle = 2* np.pi * l_rot / k_rot
        s = 1
        st = 1

        Sym_list.append(choreo.ChoreoSym(
                LoopTarget=ibody,
                LoopSource=ibody,
                SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
                TimeRev=st,
                TimeShift=fractions.Fraction(numerator=1,denominator=k_rot)
            ))


#     MomConsImposed = True
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
    save_first_init = False
    # save_first_init = True
# 
    save_all_inits = False
    # save_all_inits = True

    Save_img = True
    # Save_img = False

    Save_thumb = True
    # Save_thumb = False

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
    ncoeff_init = 300   
    # ncoeff_init = 600
    # ncoeff_init = 900
    # ncoeff_init = 1800
    # ncoeff_init = 3600
    # ncoeff_init = 1206
    # ncoeff_init = 90

    disp_scipy_opt = False
    # disp_scipy_opt = True
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = 1e-10
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
    # n_grad_change = 1.5

    coeff_ampl_o=1e-16
    k_infl=200
    k_max=800
    coeff_ampl_min=1e-16

    freq_erase_dict = 1000
    hash_dict = {}

    n_opt = 0
    # n_opt_max = 1
    # n_opt_max = 5
    n_opt_max = 1e10

    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(globals(),**locals()))
    
    choreo.Find_Choreo(**all_kwargs)
# 
# if __name__ == "__main__":
    # main(0)
# #     

if __name__ == "__main__":

    n = multiprocessing.cpu_count()
    # n = 1
    
    print(f"Executing with {n} workers")
    
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
        
        res = []
        for i in range(1,n+1):
            res.append(executor.submit(main,i))
            time.sleep(0.01)

 
