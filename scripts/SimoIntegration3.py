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
import scipy
import sys
import fractions
import functools

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','Simo_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','Simo_init_cond.txt')
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
    # SymplecticMethod = 'SymplecticGauss3'
    # SymplecticMethod = 'SymplecticGauss5' 
    SymplecticMethod = 'SymplecticGauss10'
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
    # print('#####################################')
    # print(f'    Simo {n_NT_init}')
    # print('#####################################')
    # print('')



    GoOn = True

    # i_try_max = 3
    i_try_max = 7
    # i_try_max = 15
    # i_try_max = 12
    # i_try_max = 13

    period_err_max = 1e-1
    # period_err_max = 1e-8
    period_err_wish = 1e-4

    

    IsRepresentedByFourier_wish = 1e-14
    IsRepresentedByFourier_max = 1e-9


    # CorrectAllPos = False
    # CorrectAllPos = True


    def param_to_init(params):

        x0 = np.zeros(ndof)
        v0 = np.zeros(ndof)


        x0[0] = -2 * params[0] *3
        x0[1] = 0
        x0[2] =   params[0]*3
        x0[3] =   params[1]*3
        x0[4] =   params[0]*3
        x0[5] = - params[1]*3

        v0[0] = 0
        v0[1] = -2 * params[3]
        v0[2] =   params[2]
        v0[3] =   params[3]
        v0[4] = - params[2]
        v0[5] =   params[3]

        T_NT = all_NT_init[n_NT_init,4] * 9

        phys_exp = 1/(choreo.n-1)
        rfac = (T_NT) ** phys_exp

        x0 = x0 * rfac
        v0 = v0 * rfac * T_NT 

        return (x0,v0)
    
    def init_to_param(x,v):

        T_NT = all_NT_init[n_NT_init,4] * 9

        phys_exp = 1/(choreo.n-1)
        rfac = (T_NT) ** phys_exp

        xm = x / rfac
        vm = v / (rfac * T_NT)

        params = np.zeros((4),dtype=np.float64)

        params[0] = xm[2] / 3
        params[1] = xm[3] / 3
        params[2] = vm[2]
        params[3] = vm[3]

        return params

    
    rand_param = np.random.random(4)

    xr,vr = param_to_init(rand_param)
    round_trip_param = init_to_param(xr,vr)

    assert np.linalg.norm(rand_param-round_trip_param) < 1e-10


    params_opt = all_NT_init[n_NT_init,0:4].copy()

    i_try = -1
    while(GoOn):
        i_try +=1

        # print(f'Numerical Tank {n_NT_init:4d} Try n° {i_try}')

        file_basename = 'Simo_'+(str(n_NT_init).zfill(5))
        Info_filename = os.path.join(store_folder,file_basename + '.json')

        # if os.path.isfile(Info_filename):
        #     continue

        # print("Time forward integration")
        GoOn = True
        itry = -1



        t_span = (0., 1.)
        # t_span = (0., all_NT_init[n_NT_init,4])

        # nint_anim = nbody * 1024 * 128
        # nint_sub = 16

        dnint = 16

        nint_anim = nbody * 1024 *(2**i_try)
        nint_sub = 1 

        nint = nint_sub * nint_anim
        nthird = nint // 3



        ThirdPeriodIntegrator = lambda x0, v0 : SymplecticIntegrator(
            fun = fun,
            gun = gun,
            t_span = (0., 1./3.),
            x0 = x0,
            v0 = v0,
            nint = nthird,
            keep_freq = nint_sub
        )

        def loss(params):

            x0, v0 = param_to_init(params)
            all_pos, all_v = ThirdPeriodIntegrator(x0,v0)

            xf = np.ascontiguousarray(all_pos[-1,:].copy())
            vf = np.ascontiguousarray(all_v[-1,:].copy())
            xf = xf[[2,3,4,5,0,1]].copy()
            vf = vf[[2,3,4,5,0,1]].copy()

            return init_to_param(x0-xf,v0-vf)

        params_0 = all_NT_init[n_NT_init,0:4].copy()


        period_err = np.linalg.norm(loss(params_0))

        
        print(f'Numerical Tank {n_NT_init:4d} Try n° {i_try} Error on periodicity : {period_err:.2e}')


        GoOn = (i_try < i_try_max) and  (period_err > period_err_wish)
        
# Integrate(20)
# Integrate(1)


# the_NT_init = range(len(all_NT_init))
the_NT_init = range(21,len(all_NT_init))

# the_NT_init = [0]
# the_NT_init = [4]
# the_NT_init = [18]
# the_NT_init.extend(range(25,len(all_NT_init)))

# outliers = [ 4, 18, 19, 20 ]

# 
for n_NT_init in the_NT_init:

    Integrate(n_NT_init)
# 
# for n_NT_init in the_NT_init:
# 
#     if n_NT_init not in outliers:
# 
#         Integrate(n_NT_init)

# # 
# if __name__ == "__main__":
# 
#     # n = 5
#     n = multiprocessing.cpu_count() // 2
#     # n = 4
#     # 
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