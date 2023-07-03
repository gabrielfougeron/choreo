import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

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

import datetime


def main():
    
#     nbody = 3
#     store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
#     store_folder = store_folder+str(nbody)
#     if not(os.path.isdir(store_folder)):
#         os.makedirs(store_folder)
# 

    store_folder =  os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/','fuse')


    Action_Hash_val = np.array(
 [
        13.599048114713534,
        9.140860967815005,
        11.736692511970874,
        15.991336148811968,
        23.06132952834298
    ]
    )

    rtol = 1e-5
    detect_multiples = True
    only_Action = False

    hash_dict = {}
    choreo.SelectFiles_Action(store_folder,hash_dict,Action_Hash_val=Action_Hash_val,rtol=rtol,detect_multiples=detect_multiples,only_Action=only_Action)


    # exit()
# 
#     file_basename = '00001'
#     all_pos_filename = os.path.join(store_folder,file_basename+'.npy')
#     all_pos = np.load(all_pos_filename)
#     c_coeffs = choreo.default_rfft(all_pos,axis=2,norm="forward") 
# 
#     coeff_norm = np.linalg.norm(c_coeffs,axis=(0,1))
# 
#     ncoeffs = coeff_norm.shape[0]
# 
#     for i in range(10):
#         print(i,coeff_norm[i])















if __name__ == "__main__":
    main()    
