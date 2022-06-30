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

    nbody = 5

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    duplicate_eps = 3e-1
    # duplicate_eps = 1e-8

    input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/keep/5/')
# 
    input_names_list = []
    for file_path in os.listdir(input_folder):
        file_path = os.path.join(store_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.txt' ):
            input_names_list.append(file_root)


    # input_names_list = ['00086']

    for the_name in input_names_list:

        print('')
        print(the_name)

        
        input_filename = os.path.join(input_folder,the_name)
        input_filename = input_filename + '.txt'

        input_action, input_hash = choreo.ReadActionFromFile(input_filename)

        hash_dict = {}
        file_list = choreo.SelectFiles_Action(store_folder,hash_dict,Action_val=input_action,Action_Hash_val=input_hash,rtol=duplicate_eps)

        for the_file in file_list:
            print('    ',the_file)

if __name__ == "__main__":
    main()    
