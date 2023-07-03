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
    
    # now = datetime.datetime.now()    
    # print("now =", now)

    nbody = 3

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    copy_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/copy/')

    # if os.path.isdir(copy_folder):
    #     shutil.rmtree(copy_folder)
    # 
    # os.makedirs(copy_folder)



    d_S = 1e-10

    # duplicate_eps = 2e-1
    duplicate_eps = 1e-8

#     input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/tests/')
# # 
#     input_names_list = []
#     for file_path in os.listdir(input_folder):
#         file_path = os.path.join(store_folder, file_path)
#         file_root, file_ext = os.path.splitext(os.path.basename(file_path))
#         
#         if (file_ext == '.txt' ):
#             input_names_list.append(file_root)


    rtol = 1e-5
    detect_multiples = True
    only_Action = True


    ext_list = [
        '.json',
        '.png',
        '.npy',
        # '.mp4',
        # '_thumb.png',
        ]

    input_folder = store_folder

    input_names_list = ['00002','00023']
    hash_dict = {}
    choreo.SelectFiles_Action(store_folder,hash_dict)
    hash_dict_copy = {}
    choreo.SelectFiles_Action(copy_folder,hash_dict_copy)

    file_list = []

    for key, value in hash_dict.items():

        CopyFile = True

        for the_name in input_names_list:

            input_filename = os.path.join(input_folder,the_name)
            input_filename = input_filename + '.json'

            input_hash = choreo.ReadHashFromFile(input_filename)

            # print(the_name,key, abs(value[0] - input_hash[0]))

            CopyFile = CopyFile and not( abs(value[0] - input_hash[0]) < d_S )
            # CopyFile = CopyFile & (value[0] < (input_hash[0] + d_S))

        if CopyFile:

            file_list.append(key)


    for the_file in file_list:
        
        the_hash = hash_dict[the_file]

        Already_in = False

        for key_master, val_master in hash_dict_copy.items():

            IsCandidate = choreo.TestHashSame(the_hash, val_master, rtol = rtol, detect_multiples = detect_multiples, only_Action = only_Action)

            Already_in = Already_in or IsCandidate

        if not(Already_in):
            
            print(f'Found new : {the_file}')

            for ext in ext_list:

                filename = the_file + ext
                
                source = os.path.join(store_folder,filename)
                shutil.copy(source,copy_folder)



if __name__ == "__main__":
    main()    
