import os
import concurrent.futures
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# os.environ['NUMBA_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'


import json
import shutil
import random
import time
import math as m
import numpy as np
import sys
import fractions
import functools

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','NumericalTank_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','init_data_np.txt')
all_NT_init = np.loadtxt(NT_init_filename)

def Integrate(n_NT_init):

    geodim = 2
    nbody = 3

    mass = np.ones(nbody)
    Sym_list = []

    # MomConsImposed = True
    MomConsImposed = False

    nint_small = 30
    n_reconverge_it_max_small = 0
    n_grad_change = 1.


    
    # Save_ODE_anim = True
    Save_ODE_anim = False



    ActionSyst_small = choreo.setup_changevar(geodim,nbody,nint_small,mass,n_reconverge_it_max_small,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

    # print(choreo.all_unique_SymplecticIntegrators.keys())

    # SymplecticMethod = 'SymplecticEuler'
    # SymplecticMethod = 'SymplecticStormerVerlet'
    # SymplecticMethod = 'SymplecticRuth3'
    # SymplecticMethod = 'SymplecticRuth4'
    # SymplecticMethod = 'SymplecticGauss1'
    # SymplecticMethod = 'SymplecticGauss2'
    SymplecticMethod = 'SymplecticGauss3'
    # SymplecticMethod = 'SymplecticGauss5' 
    # SymplecticMethod = 'SymplecticGauss10'
    # SymplecticMethod = 'SymplecticGauss15'

    mul_x = True
    # mul_x = False

    parallel = False

    SymplecticIntegrator = choreo.GetSymplecticIntegrator(SymplecticMethod, mul_x = mul_x)
    # SymplecticTanIntegrator = choreo.GetSymplecticTanIntegrator(SymplecticMethod)

    fun,gun = ActionSyst_small.GetSymplecticODEDef(mul_x = mul_x, parallel = parallel)
    # grad_fun,grad_gun = ActionSyst_small.GetSymplecticTanODEDef()
    ndof = nbody*ActionSyst_small.geodim

    w = np.zeros((2*ndof,2*ndof),dtype=np.float64)
    w[0:ndof,ndof:2*ndof] = np.identity(ndof)
    w[ndof:2*ndof,0:ndof] = -np.identity(ndof)

    # 
    print('')
    print('#####################################')
    print(f'    Numerical Tank {n_NT_init}')
    print('#####################################')
    print('')



    GoOn = True

    i_try_max = 10
    # i_try_max = 12
    # i_try_max = 13

    period_err_max = 1e-2
    # period_err_max = 1e-8
    period_err_wish = 1e-10

    
    IsRepresentedByFourier_wish = 1e-14
    IsRepresentedByFourier_max = 1e-9



    # CorrectAllPos = False
    CorrectAllPos = True

    i_try = -1
    while(GoOn):
        i_try +=1

    # for i_try in [7]:



        # print(f'Numerical Tank {n_NT_init:4d} Try n° {i_try}')

        file_basename = 'NumericalTank_'+(str(n_NT_init).zfill(5))
        Info_filename = os.path.join(store_folder,file_basename + '.json')

        # if os.path.isfile(Info_filename):
        #     continue

        # print("Time forward integration")
        GoOn = True
        itry = -1


        x0 = np.zeros(ndof)
        v0 = np.zeros(ndof)

        # Initial positions: (-1, 0), (1, 0), (0, 0)
        # Initial velocities: (va, vb), (va, vb), (-2vb, -2vb)

        x0[0] = -1
        x0[2] = 1

        va = all_NT_init[n_NT_init,0]
        vb = all_NT_init[n_NT_init,1]
        T_NT = all_NT_init[n_NT_init,2]
        T_NT_s = all_NT_init[n_NT_init,3]

        v0[0] = va
        v0[1] = vb
        v0[2] = va
        v0[3] = vb
        v0[4] = -2*va
        v0[5] = -2*vb

        phys_exp = 1/(choreo.n-1)
        rfac = (T_NT) ** phys_exp

        x0 = x0 * rfac
        v0 = v0 * rfac * T_NT

        t_span = (0., 1.)

        # nint_anim = nbody * 1024 * 128
        # nint_sub = 16

        dnint = 16

        nint_anim = nbody * 1024 *(2**i_try)
        nint_sub = 1 

        nint = nint_sub * nint_anim


        OnePeriodIntegrator = lambda x0, v0 : SymplecticIntegrator(
            fun = fun,
            gun = gun,
            t_span = t_span,
            x0 = x0,
            v0 = v0,
            nint = nint,
            keep_freq = nint_sub
        )

        all_pos, all_v = OnePeriodIntegrator(x0,v0)

        xf = all_pos[-1,:].copy()
        vf = all_v[-1,:].copy()

        dvx0 = np.concatenate((x0-xf,v0-vf)).reshape(2*ndof)
        period_err = np.linalg.norm(dvx0)

        all_pos[1:,:] = all_pos[:-1,:].copy()
        all_pos[0,:] = x0.copy()
        all_pos = np.ascontiguousarray(all_pos.transpose().reshape(ActionSyst_small.nbody,ActionSyst_small.geodim,nint_anim))

        if CorrectAllPos:

            choreo.InplaceCorrectPeriodicity(all_pos,x0,v0,xf,vf)


        all_coeffs_c = choreo.default_rfft(all_pos,norm="forward")


        
        ncoeff = nint // 2 + 1

        ncoeff_plotm1 = ncoeff - 1


        iprob = (ncoeff * 1) // 2
        cur_max = 0.
        for k in range(iprob):
            k_inv = ncoeff_plotm1 - k

            ampl = np.linalg.norm(all_coeffs_c[:,:,k_inv])

            cur_max = max(cur_max,ampl)


        IsRepresentedByFourier = (cur_max < IsRepresentedByFourier_wish)

        
        print(f'Numerical Tank {n_NT_init:4d} Try n° {i_try} Error on periodicity : {period_err:.2e} Fourier at probe : {cur_max:.2e}')


        # GoOn = (i_try < i_try_max) and  (period_err > period_err_wish)
        GoOn = (i_try < i_try_max) and  not(IsRepresentedByFourier)



    # if (period_err < period_err_max):
    # if (IsRepresentedByFourier):
    if (cur_max < IsRepresentedByFourier_max):


        Transform_Sym = None

        ReconvergeSol = True

        Save_All_Pos = True
        # Save_All_Pos = False

        # Save_All_Coeffs = True
        Save_All_Coeffs = False

        Save_coeff_profile = True
        # Save_coeff_profile = False

        # Save_All_Coeffs_No_Sym = True
        Save_All_Coeffs_No_Sym = False

        Save_Newton_Error = True
        # Save_Newton_Error = False

        Save_GradientAction_Error = True
        # Save_GradientAction_Error = False

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

        # Save_anim = True
        Save_anim = False


        vid_size = (8,8) # Image size in inches
        # nint_plot_anim = 2*2*2*3*3
        nint_plot_anim = nint // (3 * 8)
        dnint = 30

        nint_plot_img = nint_plot_anim * dnint

        min_n_steps_ode = 1*nint_plot_anim

        nperiod_anim = 1.

        Plot_trace_anim = True
        # Plot_trace_anim = False

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

        # save_first_init = False
        save_first_init = True

        save_all_inits = False
        # save_all_inits = True

        # max_norm_on_entry = 1e-6
        max_norm_on_entry = 1e6

        # mul_coarse_to_fine = 3
        mul_coarse_to_fine = 10

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

        Newt_err_norm_max = 1e-12
        Newt_err_norm_max_save = 1e10

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

        gradtol_list =          [1e-13  ]
        inner_maxiter_list =    [100    ]
        maxiter_list =          [3      ]
        outer_k_list =          [7      ]
        store_outer_Av_list =   [True   ]
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


        geodim = 2
        nbody = 3



        mass = np.ones(nbody)
        Sym_list = []


        # MomConsImposed = True
        MomConsImposed = False

        n_reconverge_it_max = 0
        n_grad_change = 1.

        TwoDBackend = (geodim == 2)

        GradHessBackend="Cython"
        # GradHessBackend="Numba"

        # ParallelBackend = True
        ParallelBackend = False

        nint_init = nint
        ncoeff = nint // 2 + 1

        # Do_Speed_test = True
        Do_Speed_test = False
        n_test = 3

        all_coeffs_c = choreo.default_rfft(all_pos,norm="forward")


        all_coeffs_init = np.zeros((nbody,geodim,ncoeff,2),dtype=np.float64)
        all_coeffs_init[:,:,:,0] = all_coeffs_c[:,:,:].real
        all_coeffs_init[:,:,:,1] = all_coeffs_c[:,:,:].imag

#         Sym_list.append(
#             choreo.ChoreoSym(
#                 LoopTarget=1,
#                 LoopSource=0,
#                 SpaceRot= np.array([[-1,0],[0,-1]],dtype=np.float64),
#                 TimeRev=-1,
#                 TimeShift=fractions.Fraction(
#                     numerator=0,
#                     denominator=2)
#             ))
#         Sym_list.append(
#             choreo.ChoreoSym(
#                 LoopTarget=2,
#                 LoopSource=2,
#                 SpaceRot= np.array([[-1,0],[0,-1]],dtype=np.float64),
#                 TimeRev=-1,
#                 TimeShift=fractions.Fraction(
#                     numerator=0,
#                     denominator=2)
#             ))
#         nloop = 2
# 
#         all_coeffs_init = np.zeros((nloop,geodim,ncoeff,2),dtype=np.float64)
#         all_coeffs_init[0,:,:,0] = all_coeffs_c[0,:,:].real
#         all_coeffs_init[0,:,:,1] = all_coeffs_c[0,:,:].imag
#         all_coeffs_init[1,:,:,0] = all_coeffs_c[2,:,:].real
#         all_coeffs_init[1,:,:,1] = all_coeffs_c[2,:,:].imag

        SkipCheckRandomMinDist = True

        if Do_Speed_test:

            grad_backend_list = [
                choreo.Empty_Backend_action,
                choreo.Compute_action_Cython_2D_serial,
                choreo.Compute_action_Numba_2D_serial,
                # choreo.Compute_action_Cython_nD_serial,
                # choreo.Compute_action_Numba_nD_serial,
                # choreo.Compute_action_Cython_2D_parallel,
                # choreo.Compute_action_Numba_2D_parallel,
                # choreo.Compute_action_Cython_nD_parallel,
                # choreo.Compute_action_Numba_nD_parallel,
            ]

            hess_backend_list = [
                choreo.Empty_Backend_hess_mul,
                choreo.Compute_action_hess_mul_Cython_2D_serial,
                choreo.Compute_action_hess_mul_Numba_2D_serial,
                # choreo.Compute_action_hess_mul_Cython_nD_serial,
                # choreo.Compute_action_hess_mul_Numba_nD_serial,
                # choreo.Compute_action_hess_mul_Cython_2D_parallel,
                # choreo.Compute_action_hess_mul_Numba_2D_parallel,
                # choreo.Compute_action_hess_mul_Cython_nD_parallel,
                # choreo.Compute_action_hess_mul_Numba_nD_parallel,
            ]


            for i in range(len(grad_backend_list)):

                grad_backend = grad_backend_list[i]
                hess_backend = hess_backend_list[i]

                print('')
                print(grad_backend.__name__)
                print(hess_backend.__name__)

                all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Speed_test,dict(globals(),**locals()))
                choreo.Speed_test(**all_kwargs)

        all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(globals(),**locals()))

        choreo.Find_Choreo(**all_kwargs)


    else:

        print(f'Numerical Tank {n_NT_init:4d} could not integrate. Error: {period_err} Fourier at probe : {cur_max:.2e}')


# Integrate(20)
Integrate(2)


# the_NT_init = range(len(all_NT_init))

# the_NT_init = [20]
# the_NT_init.extend(range(25,len(all_NT_init)))


# for n_NT_init in the_NT_init:
# # for n_NT_init in range(21,len(all_NT_init)):
# 
#     Integrate(n_NT_init)

# # 
# if __name__ == "__main__":
# 
#     n = 5
#     # n = multiprocessing.cpu_count() // 2
#     # n = 2
#     
#     print(f"Executing with {n} workers")
#     
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
#         
#         res = []
#         for n_NT_init in range(len(all_NT_init)):
#         # for n_NT_init in range(5):
#         # for n_NT_init in [4]:
#             res.append(executor.submit(Integrate,n_NT_init))
#             time.sleep(0.01)