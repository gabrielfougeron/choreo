import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
import json
import shutil
import random
import time
import math as m
import numpy as np
import scipy
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

twopi = 2*np.pi

def main():


    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/3/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/01 - Classic gallery')
    input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/02 - Families/02 - Chains/04')
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
# 
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

    # input_names_list = ['01 - Figure eight']
    # input_names_list = ['14 - Small mass gap']
    # input_names_list = ['09 - 3x2 Circles']
    # input_names_list = ['06 - Ten petal flower']
    # input_names_list = ['13 - 100 bodies']
    # input_names_list = ['02 - Celtic knot']
    # input_names_list = ['07 - No symmetry']
    # input_names_list = ['11 - Resonating loops']
    # input_names_list = ['10 - Complex symmetry']
    # input_names_list = ['12 - Big mass gap']
    
    input_names_list = ['1-chain']



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

    file_basename = the_name
    
    Info_filename = os.path.join(input_folder,the_name + '.json')

    with open(Info_filename,'r') as jsonFile:
        Info_dict = json.load(jsonFile)


    input_filename = os.path.join(input_folder,the_name + '.npy')

    bare_name = the_name.split('/')[-1]

    all_pos = np.load(input_filename)
    nint_init = Info_dict["n_int"]
    # ncoeff_init = Info_dict["n_Fourier"] 
    ncoeff_init = nint_init //2 + 1

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

    all_coeffs_init = np.copy(all_coeffs)


    Transform_Sym = None


    nbody = Info_dict['nbody']
    mass = np.array(Info_dict['mass']).astype(np.float64)
    Sym_list = choreo.Make_SymList_From_InfoDict(Info_dict,Transform_Sym)


    MomConsImposed = True
    # MomConsImposed = False

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

    ActionSyst = choreo.setup_changevar(nbody,nint_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

    x = ActionSyst.Package_all_coeffs(all_coeffs_init)

    ActionSyst.SavePosFFT(x)
    ActionSyst.Do_Pos_FFT = False

    Action,Gradaction = ActionSyst.Compute_action(x)
    Newt_err = ActionSyst.Compute_Newton_err(x)

    Newt_err_norm = np.linalg.norm(Newt_err)/(ActionSyst.nint*ActionSyst.nbody)

    print(f'Saved Newton Error : {Info_dict["Newton_Error"]}')
    print(f'Init Newton Error : {Newt_err_norm}')

    n_eig = 20

    # which_eigs = 'LM' # Largest (in magnitude) eigenvalues.
    # which_eigs = 'SM' # Smallest (in magnitude) eigenvalues.
    # which_eigs = 'LA' # Largest (algebraic) eigenvalues.
    # which_eigs = 'SA' # Smallest (algebraic) eigenvalues.
    which_eigs = 'BE' # Half (k/2) from each end of the spectrum.

    HessMat = ActionSyst.Compute_action_hess_LinOpt(x)
    w ,v = scipy.sparse.linalg.eigsh(HessMat,k=n_eig,which=which_eigs)
    print(w)

    n = v.shape[0]

    print(v.shape)
    # print(v[:,-1])
    
    i_eig = -1
    
    eps = 1e-9
# 
#     for i in range(n):
# 
#         if abs(v[i,i_eig]) > eps:
#             print(i,v[i,i_eig])

    vect = np.copy(v[:,i_eig])

    the_coeffs = ActionSyst.Unpackage_all_coeffs(vect)

    for k in range(ActionSyst.ncoeff):

        for il in range(ActionSyst.nloop):
            for idim in range(choreo.ndim):
                for ift in range(2):

                    val = the_coeffs[il,idim,k,ift]
# 
#                     if abs(val) > eps :
#                         print(il,idim,k,ift,val)

    the_coeffs_c = the_coeffs.view(dtype=np.complex128)[...,0]
    the_pos = choreo.the_irfft(the_coeffs_c,norm="forward")

    for iint in range(ActionSyst.nint):

        for il in range(ActionSyst.nloop):
            for idim in range(choreo.ndim):

                val = the_pos[il,idim,iint]
# 
#                 if abs(val) > eps :
#                     print(il,idim,iint,val)



if __name__ == "__main__":
    main()    
