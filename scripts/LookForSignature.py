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

import datetime


def main():
    
    now = datetime.datetime.now()    
    print("now =", now)

    nbody = 10

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    d_S = -1e-8

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


    input_folder = store_folder

    input_names_list = ['00001']
    hash_dict = {}
    choreo.SelectFiles_Action(store_folder,hash_dict)

    for the_name in input_names_list:

        print('')
        print(the_name)

        
        input_filename = os.path.join(input_folder,the_name)
        input_filename = input_filename + '.txt'

        input_action, input_hash = choreo.ReadActionFromFile(input_filename)


        file_list = []
        for key, value in hash_dict.items():

            IsCandidate = (abs(input_action-value[0]) < ((abs(input_action)+abs(value[0]))*duplicate_eps))
            for ihash in range(choreo.nhash):
                IsCandidate = (IsCandidate and ((abs(input_hash[ihash]-value[1][ihash])) < ((abs(value[1][ihash])+abs(input_hash[ihash]))*duplicate_eps)))

            # if (IsCandidate):
            #     file_list.append(key)

            if (value[0] < (input_action + d_S)):
                file_list.append(key)


        for the_file in file_list:
            print('    ',the_file)

        copy_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/copy/')
# 
#         if os.path.isdir(copy_folder):
#             shutil.rmtree(copy_folder)
        
        # os.makedirs(copy_folder)

        for the_file in file_list:

            ext_list = [
                '.txt',
                '.png',
                '.npy',
                # '.mp4',
                # '_thumb.png',
                ]

            for ext in ext_list:

                filename = the_file + ext
                
                source = os.path.join(store_folder,filename)
                shutil.copy(source,copy_folder)



if __name__ == "__main__":
    main()    
