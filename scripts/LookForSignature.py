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

def main():
    

    input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/keep/7/')
    input_filename = os.path.join(input_folder,'7_3')
    input_filename = input_filename + '.txt'

    input_action, input_hash = choreo.ReadActionFromFile(input_filename)

    file_basename = ''
    
    nbody = 7

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    duplicate_eps = 1e-8

    hash_dict = {}
    file_list = choreo.SelectFiles_Action(store_folder,hash_dict,Action_val=input_action,Action_Hash_val=input_hash,rtol=duplicate_eps)

    print(file_list)

if __name__ == "__main__":
    main()    
