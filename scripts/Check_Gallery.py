import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

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


    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/01 - Classic gallery')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/10/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/5_diff_mass/')
    input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery')
    # input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/03 - Targets/Figure eight/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/01 - Classic gallery')
    # input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/unsafe')
    # input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/02 - Families/02 - Chains/04')
    # input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/04 - Montaldi-Steckles-Gries')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Keep/tests')

    ''' Include all files in tree '''
    input_names_list = []
    input_files_list = []
    for root, dirnames, filenames in os.walk(input_folder):

        for filename in filenames:
            file_path = os.path.join(root, filename)
            file_root, file_ext = os.path.splitext(os.path.basename(file_path))

            if (file_ext == '.json' ):

                file_path = os.path.join(root, file_root)
                the_name = file_path[len(input_folder):]
                input_names_list.append(the_name)
                input_files_list.append(file_path)

# # # # 
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

    ''' Include specific files '''
    # input_names_list = ['01 - Figure eight']



    if 'input_files_list'in locals():
        Collect_files_list = not(len(input_files_list) == len(input_names_list))
    else:
        Collect_files_list = True
    
    if Collect_files_list:
        input_files_list = []
        for the_name in input_names_list:

            the_file = os.path.join(input_folder,the_name+'.json')
            if not(os.path.isfile(the_file)):
                print(the_file)
                raise ValueError('File does not exist')
            
            input_files_list.append(os.path.join(input_folder,the_name))


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
            
            for the_name, the_file in zip(input_names_list,input_files_list):

                all_kwargs = choreo.Pick_Named_Args_From_Dict(ExecName,dict(globals(),**locals()))
                res.append(executor.submit(ExecName,**all_kwargs))
                time.sleep(0.01)

    else:
            
        for the_name, the_file in zip(input_names_list,input_files_list):

            all_kwargs = choreo.Pick_Named_Args_From_Dict(ExecName,dict(globals(),**locals()))
            ExecName(the_name, the_file)


def ExecName(the_name, the_file):

    # print('--------------------------------------------')
    # print('')
    # print(the_name)
    # print('')
    # print('--------------------------------------------')
    # print('')

    file_basename = the_name


    Info_filename = the_file + '.json'

    with open(Info_filename,'r') as jsonFile:
        Info_dict = json.load(jsonFile)


    input_filename = the_file + '.npy'

    bare_name = the_name.split('/')[-1]

    all_pos = np.load(input_filename)
    nint_init = Info_dict["n_int"]

    ncoeff_init = nint_init //2 + 1

    c_coeffs = choreo.default_rfft(all_pos,axis=2,norm="forward")
    all_coeffs = np.zeros((Info_dict["nloop"],choreo.ndim,ncoeff_init,2),dtype=np.float64)
    all_coeffs[:,:,:,0] = c_coeffs.real
    all_coeffs[:,:,:,1] = c_coeffs.imag


    theta = 2*np.pi * 0.
    SpaceRevscal = 1.
    SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    TimeRev = 1.
    TimeShiftNum = 0
    TimeShiftDen = 1


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

    ActionSyst = choreo.setup_changevar(2,nbody,nint_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)



    eps = 1e-5

    ActionSyst.Center_all_coeffs(all_coeffs_init)   

    x = ActionSyst.Package_all_coeffs(all_coeffs_init)


    # print(x.shape)

    all_coeffs_round_trip = ActionSyst.Unpackage_all_coeffs(x)

    # print(np.linalg.norm(all_coeffs_round_trip - all_coeffs_init))




    ActionSyst.SavePosFFT(x)
    ActionSyst.Do_Pos_FFT = False

    Action,Gradaction = ActionSyst.Compute_action(x)
    Newt_err = ActionSyst.Compute_Newton_err(x)

    Newt_err_norm = np.linalg.norm(Newt_err)/(ActionSyst.nint*ActionSyst.nbody)

    if (Newt_err_norm > eps):

        print('')
        print(the_name)

        print(f'Saved Grad Action : {Info_dict["Grad_Action"]}')
        print(f'Init Grad Action : {np.linalg.norm(Gradaction)}')

        print(f'Saved Newton Error : {Info_dict["Newton_Error"]}')
        print(f'Init Newton Error : {Newt_err_norm}')


#     print('')
#     print(the_name)
# 
#     print(f'Saved Grad Action : {Info_dict["Grad_Action"]}')
#     print(f'Init Grad Action : {np.linalg.norm(Gradaction)}')
# # 
#     print(f'Saved Newton Error : {Info_dict["Newton_Error"]}')
#     print(f'Init Newton Error : {Newt_err_norm}')

    return


if __name__ == "__main__":
    main()    
