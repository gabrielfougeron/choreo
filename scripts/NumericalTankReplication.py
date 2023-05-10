import os
import concurrent.futures
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


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
import functools
import inspect

import tqdm

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','NumericalTank_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','init_data_np.txt')
all_NT_init = np.loadtxt(NT_init_filename)



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

Save_anim = True
# Save_anim = False


vid_size = (8,8) # Image size in inches
# nint_plot_anim = 2*2*2*3*3
nint_plot_anim = 2*2*2*3*3*5
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
Newt_err_norm_max_save = 1e-9

# krylov_method = 'lgmres'
krylov_method = 'gmres'
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

gradtol_list =          [1e-13  ,1e-13  ,1e-13  ]
inner_maxiter_list =    [100    ,100    ,100    ]
maxiter_list =          [1      ,20     ,1      ]
outer_k_list =          [7      ,7      ,7      ]
store_outer_Av_list =   [True   ,True   ,True   ]
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

nint_first_try = nbody * 128


mass = np.ones(nbody)
Sym_list = []


# Sym_list.append(
#     choreo.ChoreoSym(
#         LoopTarget=1,
#         LoopSource=0,
#         SpaceRot= np.array([[-1,0],[0,-1]],dtype=np.float64),
#         TimeRev=-1,
#         TimeShift=fractions.Fraction(
#             numerator=0,
#             denominator=2)
#     ))
# Sym_list.append(
#     choreo.ChoreoSym(
#         LoopTarget=2,
#         LoopSource=2,
#         SpaceRot= np.array([[-1,0],[0,-1]],dtype=np.float64),
#         TimeRev=-1,
#         TimeShift=fractions.Fraction(
#             numerator=0,
#             denominator=2)
#     ))



# MomConsImposed = True
MomConsImposed = False

n_reconverge_it_max = 1
n_grad_change = 1.

TwoDBackend = (geodim == 2)

GradHessBackend="Cython"
# GradHessBackend="Numba"

# ParallelBackend = True
ParallelBackend = False

nint_small = 30
n_reconverge_it_max_small = 0

Do_Speed_test = False
# Do_Speed_test = True

ActionSyst_small = choreo.setup_changevar(geodim,nbody,nint_small,mass,n_reconverge_it_max_small,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

# print(choreo.all_unique_SymplecticIntegrators.keys())

# SymplecticMethod = 'SymplecticEuler'
# SymplecticMethod = 'SymplecticStormerVerlet'
# SymplecticMethod = 'SymplecticRuth3'
# SymplecticMethod = 'SymplecticRuth4'
# SymplecticMethod = 'SymplecticGauss1'
# SymplecticMethod = 'SymplecticGauss2'
# SymplecticMethod = 'SymplecticGauss3'
# SymplecticMethod = 'SymplecticGauss5' 
SymplecticMethod = 'SymplecticGauss10'

SymplecticIntegrator = choreo.GetSymplecticIntegrator(SymplecticMethod)
SymplecticTanIntegrator = choreo.GetSymplecticTanIntegrator(SymplecticMethod)

# disp_scipy_opt = False
disp_scipy_opt = True


# nint_ODE_mul = 64
# nint_ODE_mul =  2**11
# nint_ODE_mul =  2**7
# nint_ODE_mul =  2**5
# nint_ODE_mul =  2**3
# nint_ODE_mul =  2**1
nint_ODE_mul =  1


fun,gun = ActionSyst_small.GetSymplecticODEDef()
grad_fun,grad_gun = ActionSyst_small.GetSymplecticTanODEDef()
ndof = nbody*ActionSyst_small.geodim

w = np.zeros((2*ndof,2*ndof),dtype=np.float64)
w[0:ndof,ndof:2*ndof] = np.identity(ndof)
w[ndof:2*ndof,0:ndof] = -np.identity(ndof)

for n_NT_init in [0]:
# for n_NT_init in [4]:
# for n_NT_init in range(len(all_NT_init)):
# for n_NT_init in range(4,len(all_NT_init)):


    file_basename = 'NumericalTank_'+(str(n_NT_init).zfill(5))
    Info_filename = os.path.join(store_folder,file_basename + '.json')

    # if os.path.isfile(Info_filename):
    #     continue


    print("Time forward integration")
    GoOn = True
    itry = -1

    while GoOn:
    # for i in range(1):



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

        # print(x0,v0)


        ndof = nbody * geodim

        grad_x0 = np.zeros((ndof,2*ndof),dtype=np.float64)
        grad_v0 = np.zeros((ndof,2*ndof),dtype=np.float64)
        for idof in range(ndof):
            grad_x0[idof,idof] = 1
        for idof in range(ndof):
            grad_v0[idof,ndof+idof] = 1




        t_span = (0., 1.)

        itry += 1
        nint = nint_first_try * (2**itry)

        n_ODE = nint*nint_ODE_mul

        OnePeriodIntegrator = lambda x0, v0 : SymplecticIntegrator(
                fun = fun,
                gun = gun,
                t_span = t_span,
                x0 = x0,
                v0 = v0,
                nint = n_ODE,
                keep_freq = n_ODE)

        OnePeriodTanIntegrator = lambda x0, v0, grad_x0, grad_v0 : SymplecticTanIntegrator(
                fun = fun,
                gun = gun,
                grad_fun = grad_fun,
                grad_gun = grad_gun,
                t_span = t_span,
                x0 = x0,
                v0 = v0,
                grad_x0 = grad_x0,
                grad_v0 = grad_v0,
                nint = n_ODE,
                keep_freq = n_ODE)



        print('')
        print(f'nint = {nint}')
        print(f'nODE = {n_ODE}')
    

        loss = functools.partial(choreo.ComputePeriodicityDefault, OnePeriodIntegrator = OnePeriodIntegrator)
        grad_loss = functools.partial(choreo.ComputeGradPeriodicityDefault, OnePeriodTanIntegrator = OnePeriodTanIntegrator)
        grad_loss_mul = functools.partial(choreo.ComputeGradMulPeriodicityDefault, OnePeriodTanIntegrator = OnePeriodTanIntegrator)
        grad_only = lambda x : grad_loss(x)[1] - np.identity(2*ndof)
        grad_only_mul = lambda x,v : grad_loss_mul(x,v)[1] - v












        vx0 = np.concatenate((x0,v0)).reshape(2*ndof)
        tbeg = time.perf_counter()
        dvx0 = loss(vx0)
        best_sol = choreo.current_best(vx0,dvx0)
        tend = time.perf_counter()
        print(f'Integration time: {tend-tbeg}')


        period_err = np.linalg.norm(dvx0)
        print(f'Error on Periodicity before optimization: {period_err}')
        if (period_err > 1e-5):
            continue








        
#         dxb = np.random.random(vx0.shape)
#         # innz = 0
#         # dxb = np.zeros(vx0.shape)
#         # dxb[innz] = 1
# 
# 
#         # tbeg = time.perf_counter()
#         # all_pos, all_v = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint*nint_ODE_mul,nint_ODE_mul)
#         # tend = time.perf_counter()
#         # print(f'Integration time: {tend-tbeg}')
# 
#         tbeg = time.perf_counter()
#         exgrad = grad_only_mul(vx0,dxb)
#         tend = time.perf_counter()
#         # print(f'1 dir time: {tend-tbeg}')
# 
#         tbeg = time.perf_counter()
#         gradmat = grad_only(vx0)
#         tend = time.perf_counter()
#         # print(f'Full Mat time: {tend-tbeg}')
# 
#         eigvals,eigvects = scipy.linalg.eig(a=gradmat)
#         print('Max Eigenvalue of the Monodromy matrix :',np.abs(eigvals).max())
# 
#         exgrad_mat = np.dot(gradmat,dxb)
# 
#         print(np.linalg.norm(exgrad - exgrad_mat))
# 
#         # print(exgrad)
# 
#         # exponent_eps_list = range(16)
#         exponent_eps_list = [8]
# 
#         for exponent_eps in exponent_eps_list:
#             
#             eps = 10**(-exponent_eps)
# 
#             xp = np.copy(vx0) + eps*dxb
#             fp = loss(xp)
#             
#             xm = np.copy(vx0)
#             fm = loss(xm)
# 
#             df_difffin = (fp-fm)/(eps)
#             
#             # print(df_difffin)
#             
#             print('')
#             print('eps : ',eps)
#             err_vect = df_difffin - exgrad
#             # print('DF : ',np.linalg.norm(df_difffin))
#             # print('EX : ',np.linalg.norm(exgrad))
# 
#             Abs_diff = np.linalg.norm(err_vect)
#             print('Abs_diff : ',Abs_diff)
#             Rel_diff = np.linalg.norm(err_vect)/(np.linalg.norm(df_difffin)+np.linalg.norm(exgrad))
#             print('Rel_diff : ',Rel_diff)
# 
# 
#         exit()






        # line_search = 'armijo'
        # line_search = 'wolfe'
        line_search = None

        # linesearch_smin = 0.1
        linesearch_smin = 1.

        # Use_exact_Jacobian_T = True
        Use_exact_Jacobian_T = False

        maxiter_period_opt = 30 

        # krylov_method_T = 'lgmres'
        krylov_method_T = 'gmres'
        # krylov_method_T = 'bicgstab'
        # krylov_method_T = 'cgs'
        # krylov_method_T = 'minres'
        # krylov_method_T = 'tfqmr'

        jac_options = {'method':krylov_method_T,'rdiff':None,'inner_tol':0,'inner_M':None }

        if (Use_exact_Jacobian_T):
            jacobian = choreo.ExactKrylovJacobian(exactgrad=grad_only_mul,**jac_options)

        else: 
            jacobian = scipy.optimize.nonlin.KrylovJacobian(**jac_options)


        opt_result , info = choreo.nonlin_solve_pp(F=loss,x0=vx0,jacobian=jacobian,verbose=True,maxiter=maxiter_period_opt,f_tol=1e-15,line_search=line_search,raise_exception=False,smin=linesearch_smin,full_output=True,callback=best_sol.update)
        
        print(opt_result)
        print(info)
# 
#         vx0 = best_sol.x
#         dvx0 = best_sol.f



        # res = scipy.optimize.root(loss, vx0, method='lm', jac=grad_only, tol=1e-15,options={'maxiter':100})

        # res = scipy.optimize.root(loss, vx0, method='hybr', jac=grad_only, tol=1e-15,options={'maxfev':100})


        # print(res.message)
        # vx0 = res.x
        # dvx0 = res.fun


        period_err = np.linalg.norm(dvx0)
        print(f'Error on Periodicity after optimization: {period_err}')


        x0 = vx0[0:ndof].copy()
        v0 = vx0[ndof:2*ndof].copy()






        # tbeg = time.perf_counter()
        all_pos, all_v = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint*nint_ODE_mul,nint_ODE_mul)
        # tend = time.perf_counter()
        # print(f'Integration time: {tend-tbeg}')

        xf = all_pos[-1,:].copy()
        vf = all_v[-1,:].copy()


        # tbeg = time.perf_counter()
        # all_pos, all_v, all_grad_pos, all_grad_v = SymplecticTanIntegrator(fun,gun,grad_fun,grad_gun,t_span,x0,v0,grad_x0,grad_v0,nint*nint_ODE_mul,nint_ODE_mul)
        # tend = time.perf_counter()
        # print(f'Integration time: {tend-tbeg}')


#         grad_xf = all_grad_pos[-1,:,:].copy()
#         grad_vf = all_grad_v[-1,:,:].copy()
# 
#         MonodromyMat = np.ascontiguousarray(np.concatenate((grad_xf,grad_vf),axis=0).reshape(2*ndof,2*ndof))
# 
#         print('Symplecticity')
#         print(np.linalg.norm(w - np.dot(MonodromyMat.transpose(),np.dot(w,MonodromyMat))))
#         eigvals,eigvects = scipy.linalg.eig(a=MonodromyMat)
#         # print(eigvals)
#         # print(eigvals.real)
#         print(np.abs(eigvals))
#         # print(eigvects)


# 
#         period_err = np.linalg.norm(np.concatenate((x0-xf,v0-vf)).reshape(2*ndof))
# 
#         print(f'Error on Periodicity: {period_err}')



        all_pos[1:,:] = all_pos[:-1,:].copy()
        all_pos[0,:] = x0.copy()


        all_pos = all_pos.transpose().reshape(ActionSyst_small.nbody,ActionSyst_small.geodim,nint)
        # all_v = all_v.transpose().reshape(ActionSyst_small.nbody,ActionSyst_small.geodim,nint)


        nint_init = nint
        ncoeff_init = nint_init // 2 +1

        c_coeffs = choreo.the_rfft(all_pos,axis=2,norm="forward")
        all_coeffs = np.zeros((nbody,geodim,ncoeff_init,2),dtype=np.float64)
        all_coeffs[:,:,:,0] = c_coeffs.real
        all_coeffs[:,:,:,1] = c_coeffs.imag

        eps = 1e-12
        # eps = 0.

        ampl = np.zeros((ncoeff_init),dtype=np.float64)
        for k in range(ncoeff_init):
            ampl[k] = np.linalg.norm(c_coeffs[:,:,k])

        max_ampl = np.zeros((ncoeff_init),dtype=np.float64)

        ncoeff_plotm1 = ncoeff_init - 1

        cur_max = 0.
        for k in range(ncoeff_init):
            k_inv = ncoeff_plotm1 - k

            cur_max = max(cur_max,ampl[k_inv])
            max_ampl[k_inv] = cur_max

        iprob = (ncoeff_init * 2) // 3
        GoOn = (max_ampl[iprob] > eps) or (period_err > 1e-10)

        print(f"Max amplitude at probe index: {max_ampl[iprob]}")
        



    # exit()



    # theta = 2*np.pi * 0.
    # SpaceRevscal = 1.
    # SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    # TimeRev = 1.
    # TimeShiftNum = 0
    # TimeShiftDen = 1

    # nloop = 2
    # all_coeffs_init = np.zeros((nloop,geodim,ncoeff_init,2),dtype=np.float64)
    # all_coeffs_init[0,:,:,:] = all_coeffs[0,:,:,:]
    # all_coeffs_init[1,:,:,:] = all_coeffs[2,:,:,:]

    all_coeffs_init = all_coeffs

    # theta = 2*np.pi * 0/2
    # SpaceRevscal = 1.
    # SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    # TimeRev = 1.
    # TimeShiftNum = 0
    # TimeShiftDen = 2

    # all_coeffs_init = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)

    # all_coeffs_init = all_coeffs

    # Transform_Sym = choreo.ChoreoSym(SpaceRot=SpaceRot, TimeRev=TimeRev, TimeShift = fractions.Fraction(numerator=TimeShiftNum,denominator=TimeShiftDen))

    if Do_Speed_test:

        n_test = 10

        grad_backend_list = [
            choreo.Empty_Backend_action,
            choreo.Compute_action_Cython_2D_serial,
            choreo.Compute_action_Numba_2D_serial,
            choreo.Compute_action_Cython_nD_serial,
            choreo.Compute_action_Numba_nD_serial,
            choreo.Compute_action_Cython_2D_parallel,
            choreo.Compute_action_Numba_2D_parallel,
            choreo.Compute_action_Cython_nD_parallel,
            choreo.Compute_action_Numba_nD_parallel,
        ]

        hess_backend_list = [
            choreo.Empty_Backend_hess_mul,
            choreo.Compute_action_hess_mul_Cython_2D_serial,
            choreo.Compute_action_hess_mul_Numba_2D_serial,
            choreo.Compute_action_hess_mul_Cython_nD_serial,
            choreo.Compute_action_hess_mul_Numba_nD_serial,
            choreo.Compute_action_hess_mul_Cython_2D_parallel,
            choreo.Compute_action_hess_mul_Numba_2D_parallel,
            choreo.Compute_action_hess_mul_Cython_nD_parallel,
            choreo.Compute_action_hess_mul_Numba_nD_parallel,
        ]


        for i in range(len(grad_backend_list)):

            grad_backend = grad_backend_list[i]
            hess_backend = hess_backend_list[i]

            print('')
            print(grad_backend.__name__)
            print(hess_backend.__name__)

            all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Speed_test,dict(globals(),**locals()))
            choreo.Speed_test(**all_kwargs)


    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(**locals()))
    choreo.Find_Choreo(**all_kwargs)
    

#     try:
# 
#         all_pos = np.load(os.path.join(store_folder,file_basename+'.npy'))
# 
#         
# 
#         
# 
#     except Exception:
#         pass



