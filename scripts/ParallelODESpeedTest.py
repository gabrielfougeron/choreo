import os
import concurrent.futures
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
# os.environ['OMP_NUM_THREADS'] = str('4')
# os.environ['OMP_NUM_THREADS'] = str('1')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import json
import shutil
import random
import time
import math as m
import numpy as np
import scipy
import scipy.linalg
import sys
import functools
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime

One_sec = 1e9

twopi = 2*np.pi

def main():

    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/3/')
    input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/01 - Classic gallery')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Keep/tests')
    
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
# # 
# # # 
#     ''' Include all files in folder '''
#     input_names_list = []
#     for file_path in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, file_path)
#         file_root, file_ext = os.path.splitext(os.path.basename(file_path))
#         
#         if (file_ext == '.json' ):
#             # 
#             # if int(file_root) > 8:
#             #     input_names_list.append(file_root)
# 
#             input_names_list.append(file_root)

    input_names_list = []
    # input_names_list.append('01 - Figure eight'     )
    # input_names_list.append('14 - Small mass gap'   )
    # input_names_list.append('03 - Trefoil'          )
    # input_names_list.append('04 - 5 pointed star'   ) 
    # input_names_list.append('07 - No symmetry'   ) 
    # input_names_list.append('09 - 3x2 Circles'   ) 
    # input_names_list.append('11 - Resonating loops'   ) 
    input_names_list.append('12 - 100 bodies'   ) 






    store_folder = os.path.join(__PROJECT_ROOT__,'Reconverged_sols')
    # store_folder = input_folder

    # Exec_Mul_Proc = True
    Exec_Mul_Proc = False

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

    geodim = 2

    file_basename = the_name
    
    Info_filename = os.path.join(input_folder,the_name + '.json')

    with open(Info_filename,'r') as jsonFile:
        Info_dict = json.load(jsonFile)


    input_filename = os.path.join(input_folder,the_name + '.npy')

    bare_name = the_name.split('/')[-1]

    all_pos = np.load(input_filename)
    nint = Info_dict["n_int"]
    ncoeff_init = nint // 2 + 1

    c_coeffs = choreo.default_rfft(all_pos,axis=2,norm="forward")
    all_coeffs = np.zeros((Info_dict["nloop"],geodim,ncoeff_init,2),dtype=np.float64)
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



    # all_coeffs_init = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)
    # Transform_Sym = choreo.ChoreoSym(SpaceRot=SpaceRot, TimeRev=TimeRev, TimeShift = fractions.Fraction(numerator=TimeShiftNum,denominator=TimeShiftDen))

    all_coeffs_init = np.copy(all_coeffs)
    Transform_Sym = None


    nbody = Info_dict['nbody']
    mass = np.array(Info_dict['mass']).astype(np.float64)
    Sym_list = choreo.Make_SymList_From_InfoDict(Info_dict,Transform_Sym)


    # MomConsImposed = True
    MomConsImposed = False
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

    n_reconverge_it_max = 0
    n_grad_change = 1.

    ActionSyst = choreo.setup_changevar(2,nbody,nint,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

    x = ActionSyst.Package_all_coeffs(all_coeffs_init)

    ActionSyst.SavePosFFT(x)

    Action,Gradaction = ActionSyst.Compute_action(x)
    Newt_err = ActionSyst.Compute_Newton_err(x)

    Newt_err_norm = np.linalg.norm(Newt_err)/(ActionSyst.nint*ActionSyst.nbody)

    print(f'Saved Newton Error : {Info_dict["Newton_Error"]}')
    print(f'Init Newton Error : {Newt_err_norm}')

    ncoeff = ActionSyst.ncoeff
    nint = ActionSyst.nint
    ndof = nbody*ActionSyst.geodim

    fun_serial,gun_serial = ActionSyst.GetSymplecticODEDef(mul_x = False, parallel=False)
    fun_mul,gun_mul = ActionSyst.GetSymplecticODEDef(mul_x = True, parallel=False)
    fun_parallel,gun_parallel = ActionSyst.GetSymplecticODEDef(mul_x = True, parallel=True)

    x0, v0 = ActionSyst.Compute_init_pos_and_vel(x)
    z0 = np.ascontiguousarray(np.concatenate((x0, v0),axis=0).reshape(2*ndof))
    
    grad_fun_serial,grad_gun_serial = ActionSyst.GetSymplecticTanODEDef(mul_x = False, parallel=False)
    grad_fun_mul,grad_gun_mul = ActionSyst.GetSymplecticTanODEDef(mul_x = True, parallel=False)
    grad_fun_parallel,grad_gun_parallel = ActionSyst.GetSymplecticTanODEDef(mul_x = True, parallel=True)

    grad_x0 = np.zeros((ndof,2*ndof),dtype=np.float64)
    grad_v0 = np.zeros((ndof,2*ndof),dtype=np.float64)
    for idof in range(ndof):
        grad_x0[idof,idof] = 1
    for idof in range(ndof):
        grad_v0[idof,ndof+idof] = 1



    w = np.zeros((2*ndof,2*ndof),dtype=np.float64)
    w[0:ndof,ndof:2*ndof] = np.identity(ndof)
    w[ndof:2*ndof,0:ndof] = -np.identity(ndof)

    nint = 1000

    T = 1./1024

    # nint_ODE_mul = 64
    # nint_ODE_mul =  2**11
    # nint_ODE_mul =  2**8
    # nint_ODE_mul =  2**6
    # nint_ODE_mul =  2**3
    # nint_ODE_mul =  2**1
    nint_ODE_mul =  1

    nsteps = 8

    SymplecticMethod = 'SymplecticGauss'+str(nsteps)


    # SymplecticMethod = 'LobattoIIIA_3'                 
    # SymplecticMethod = 'LobattoIIIB_3'                 
    # SymplecticMethod = 'LobattoIIIA_4'                 
    # SymplecticMethod = 'LobattoIIIB_4'                 
    # SymplecticMethod = 'PartitionedLobattoIII_AX_BV_3' 
    # SymplecticMethod = 'PartitionedLobattoIII_AV_BX_3' 
    # SymplecticMethod = 'PartitionedLobattoIII_AX_BV_4' 
    # SymplecticMethod = 'PartitionedLobattoIII_AV_BX_4' 

    print('')
    print(f'Integration method : {SymplecticMethod}')
    print('')
    print('Direct integration')
    print('')


    SymplecticIntegrator_one_x = choreo.GetSymplecticIntegrator(SymplecticMethod, mul_x = False)
    SymplecticIntegrator_mul_x = choreo.GetSymplecticIntegrator(SymplecticMethod, mul_x = True)

    tbeg = time.perf_counter()
    all_x_serial, all_v_serial = SymplecticIntegrator_one_x(
        fun = fun_serial,
        gun = gun_serial,
        t_span = (0.,T),
        x0 = x0,
        v0 = v0,
        nint = nint*nint_ODE_mul,
        keep_freq = nint_ODE_mul
    )
    tend = time.perf_counter()

    t_serial = tend-tbeg
    print(f'Serial integration time : {t_serial}')

    tbeg = time.perf_counter()
    all_x_mul, all_v_mul = SymplecticIntegrator_mul_x(
        fun = fun_mul,
        gun = gun_mul,
        t_span = (0.,T),
        x0 = x0,
        v0 = v0,
        nint = nint*nint_ODE_mul,
        keep_freq = nint_ODE_mul
    )
    tend = time.perf_counter()

    t_mul = tend-tbeg
    print(f'Mul x integration time : {t_mul}')


    tbeg = time.perf_counter()
    all_x_parallel, all_v_parallel = SymplecticIntegrator_mul_x(
        fun = fun_parallel,
        gun = gun_parallel,
        t_span = (0.,T),
        x0 = x0,
        v0 = v0,
        nint = nint*nint_ODE_mul,
        keep_freq = nint_ODE_mul
    )
    tend = time.perf_counter()

    t_parallel = tend-tbeg
    print(f'Parallel integration time : {t_parallel}')
    print('')

    print(f'Difference btw outputs : {np.linalg.norm(all_x_serial-all_x_parallel) + np.linalg.norm(all_v_serial-all_v_parallel)}')
    print(f'Difference btw outputs : {np.linalg.norm(all_x_serial-all_x_mul) + np.linalg.norm(all_v_serial-all_v_mul)}')
    print('')

    if nsteps > 1:

        t_overhead = (t_serial - t_mul) / (nsteps - 1)
        t_bare = (nsteps * t_mul - t_serial ) / (nsteps - 1)

        print(f't_bare : {t_bare}')
        print(f't_overhead : {t_overhead}')


    print('')
    print('Tangent integration')
    print('')

    SymplecticTanIntegrator_one_x = choreo.GetSymplecticTanIntegrator(SymplecticMethod, mul_x = False)
    SymplecticTanIntegrator_mul_x = choreo.GetSymplecticTanIntegrator(SymplecticMethod, mul_x = True)


    tbeg = time.perf_counter()
    all_x_serial, all_v_serial, all_grad_x_serial, all_grad_v_serial = SymplecticTanIntegrator_one_x(
        fun = fun_serial,
        gun = gun_serial,
        grad_fun = grad_fun_serial,
        grad_gun = grad_gun_serial,
        t_span = (0.,T),
        x0 = x0,
        v0 = v0,
        grad_x0 = grad_x0,
        grad_v0 = grad_v0,
        nint = nint*nint_ODE_mul,
        keep_freq = nint_ODE_mul
    )
    tend = time.perf_counter()

    t_serial = tend-tbeg
    print(f'Serial Tan integration time : {t_serial}')

    tbeg = time.perf_counter()
    all_x_mul, all_v_mul, all_grad_x_mul, all_grad_v_mul = SymplecticTanIntegrator_mul_x(
        fun = fun_mul,
        gun = gun_mul,
        grad_fun = grad_fun_mul,
        grad_gun = grad_gun_mul,
        t_span = (0.,T),
        x0 = x0,
        v0 = v0,
        grad_x0 = grad_x0,
        grad_v0 = grad_v0,
        nint = nint*nint_ODE_mul,
        keep_freq = nint_ODE_mul
    )
    tend = time.perf_counter()

    t_mul = tend-tbeg
    print(f'Mul x Tan integration time : {t_mul}')


    tbeg = time.perf_counter()
    all_x_parallel, all_v_parallel, all_grad_x_parallel, all_grad_v_parallel = SymplecticTanIntegrator_mul_x(
        fun = fun_parallel,
        gun = gun_parallel,
        grad_fun = grad_fun_parallel,
        grad_gun = grad_gun_parallel,
        t_span = (0.,T),
        x0 = x0,
        v0 = v0,
        grad_x0 = grad_x0,
        grad_v0 = grad_v0,
        nint = nint*nint_ODE_mul,
        keep_freq = nint_ODE_mul
    )
    tend = time.perf_counter()

    t_parallel = tend-tbeg
    print(f'Parallel Tan integration time : {t_parallel}')

    print('')

    err = np.linalg.norm(all_x_serial-all_x_parallel) + np.linalg.norm(all_v_serial-all_v_parallel) + np.linalg.norm(all_grad_x_serial-all_grad_x_parallel) + np.linalg.norm(all_grad_v_serial-all_grad_v_parallel) 
    print(f'Difference btw outputs : {err}')

    # print(f'Difference btw outputs : {np.linalg.norm(all_x_serial-all_x_parallel)}')
    # print(f'Difference btw outputs : {np.linalg.norm(all_v_serial-all_v_parallel) }')
    # print(f'Difference btw outputs : {np.linalg.norm(all_grad_x_serial-all_grad_x_parallel)}')
    # print(f'Difference btw outputs : {np.linalg.norm(all_grad_v_serial-all_grad_v_parallel) }')

    err = np.linalg.norm(all_x_serial-all_x_mul) + np.linalg.norm(all_v_serial-all_v_mul) + np.linalg.norm(all_grad_x_serial-all_grad_x_mul) + np.linalg.norm(all_grad_v_serial-all_grad_v_mul) 
    print(f'Difference btw outputs : {err}')

    print('')
    
    if nsteps > 1:

        t_overhead = (t_serial - t_mul) / (nsteps - 1)
        t_bare = (nsteps * t_mul - t_serial ) / (nsteps - 1)

        print(f't_bare : {t_bare}')
        print(f't_overhead : {t_overhead}')


#     tbeg = time.perf_counter()
# 
#     all_x, all_v, all_grad_x, all_grad_v = SymplecticTanIntegrator(
#         fun = fun,
#         gun = gun,
#         grad_fun = grad_fun,
#         grad_gun = grad_gun,
#         t_span = (0.,T),
#         x0 = x0,
#         v0 = v0,
#         grad_x0 = grad_x0,
#         grad_v0 = grad_v0,
#         nint = nint*nint_ODE_mul,
#         keep_freq = nint_ODE_mul
#     )
# 
#     tend = time.perf_counter()
# 
#     print(f'CPU time of integration :{tend-tbeg}')
# 
# 
#     xf = all_x[-1,:].copy()
#     vf = all_v[-1,:].copy()
#     zf = np.ascontiguousarray(np.concatenate((xf, vf),axis=0).reshape(2*ndof))
# 
#     period_err = np.linalg.norm(zf-z0)
#     print(f'Error on Periodicity: {period_err}')
# 
# 
#     grad_xf = all_grad_x[-1,:,:].copy()
#     grad_vf = all_grad_v[-1,:,:].copy()
# 
#     MonodromyMat = np.ascontiguousarray(np.concatenate((grad_xf,grad_vf),axis=0).reshape(2*ndof,2*ndof))
# 
#     # MonodromyMat = np.ascontiguousarray(MonodromyMat.T)
# 
#     # print(MonodromyMat)
# # 
#     print('Symplecticity')
#     print(np.linalg.norm(w - np.dot(MonodromyMat.transpose(),np.dot(w,MonodromyMat))) / (np.linalg.norm(MonodromyMat)**2))
#     # print((w - np.dot(MonodromyMat.transpose(),np.dot(w,MonodromyMat))))
# 
# 
# 
# 
#     eigvals,eigvects = scipy.linalg.eig(a=MonodromyMat)
#     print('Max Eigenvalue of the Monodromy matrix :',np.abs(eigvals).max())
#     # print('Eigenvalues of the Monodromy matrix :')
#     # print(eigvals)
#     # print(eigvals.real)
#     # print(np.abs(eigvals))
#     # print(eigvects)
# 
# 
#     # print(MonodromyMatLog)
#     # print(MonodromyMat)
# 
#     # '''Eigendecomposition'''
#     # Instability_magnitude,Instability_directions = choreo.InstabilityDecomposition(MonodromyMat)
# 
#     # print(Instability_magnitude)
# 
#     # Evaluates the relative accuracy of the Monodromy matrix integration process
#     # f0 should be an eigenvector of the Monodromy matrix, with eigenvalue 1
#     z0 = np.ascontiguousarray(np.concatenate((x0, v0),axis=0).reshape(2*ndof))
#     f0 = ActionSyst.Compute_Auto_ODE_RHS(z0)
#     print(f'Relative error on flow eigenstate: {np.linalg.norm(MonodromyMat.dot(f0)-f0)/np.linalg.norm(f0):e}')
#     # print(the_name+f' {Instability_magnitude[:]}')
#     # print(the_name+f' {np.flip(1/Instability_magnitude[:])}')
#     # print("Relative error on loxodromy ",np.linalg.norm(Instability_magnitude - np.flip(1/Instability_magnitude))/np.linalg.norm(Instability_magnitude))
# 
# 
# 
# 










if __name__ == "__main__":
    main()    
