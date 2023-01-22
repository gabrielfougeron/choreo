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
import scipy.integrate

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime


def main():

    input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/mod')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/10/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/copy/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/keep/13')

#     ''' Include all files in tree '''
#     input_names_list = []
#     for root, dirnames, filenames in os.walk(input_folder):
# 
#         for filename in filenames:
#             file_path = os.path.join(root, filename)
#             file_root, file_ext = os.path.splitext(os.path.basename(file_path))
# 
#             if (file_ext == '.txt' ):
# 
#                 file_path = os.path.join(root, file_root)
#                 the_name = file_path[len(input_folder):]
#                 input_names_list.append(the_name)


    ''' Include all files in folder '''
    input_names_list = []
    for file_path in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.txt' ):
            # input_names_list.append(file_root)
            input_names_list.append(file_root+'_nosym')

    # input_names_list = ['00006']
    # input_names_list = ['00006_nosym']

    GradActionThresh = 1e-8

    for the_name in input_names_list:

        print('')
        print(the_name)

        input_filename = os.path.join(input_folder,the_name)
        input_filename = input_filename + '.npy'

        bare_name = the_name.split('/')[-1]

        all_coeffs = np.load(input_filename)

        theta = 2*np.pi * 0.5
        SpaceRevscal = 1.
        SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
        TimeRev = 1.
        TimeShiftNum = 0
        TimeShiftDen = 2

        all_coeffs = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)

        ncoeff_init = all_coeffs.shape[2]

        the_i = -1
        the_i_max = 0

        Gradaction_OK = False

        while (not(Gradaction_OK) and (the_i < the_i_max)):

            the_i += 1



            p = 1
            # p_list = range(the_i_max)
            # p_list = [3]
            # p = p_list[the_i%len(p_list)]

            nc = 3

            mm = 1
            # mm_list = [1]
            # mm = mm_list[the_i%len(mm_list)]

            # nbpl=[nc]
            nbpl=[1 for i in range(nc)]
# 
#             SymType = {
#                 'name'  : 'D',
#                 'n'     : nc,
#                 'm'     : mm,
#                 'l'     : 0,
#                 'k'     : 1,
#                 'p'     : p,
#                 'q'     : nc,
#             }
#             Sym_list = choreo.Make2DChoreoSym(SymType,range(nc))
#             nbody = nc

            Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymName='C')

            mass = np.ones((nbody),dtype=np.float64)

            # MomConsImposed = True
            MomConsImposed = False

            n_reconverge_it_max = 0
            n_grad_change = 1.

            ActionSyst = choreo.setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

            x = ActionSyst.Package_all_coeffs(all_coeffs)

            Action,Gradaction = ActionSyst.Compute_action(x)

            Gradaction_OK = (np.linalg.norm(Gradaction) < GradActionThresh)

        if not(Gradaction_OK):
            raise(ValueError('Correct Symmetries not found'))

        n_eig = 10

        HessMat = ActionSyst.Compute_action_hess_LinOpt(x)
        w ,v = scipy.sparse.linalg.eigsh(HessMat,k=n_eig,which='SA')
        print(w)


if __name__ == "__main__":
    main()    
