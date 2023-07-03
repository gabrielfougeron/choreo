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

    ext_list = [
        '.json',
        '.png',
        '.npy',
        # '.mp4',
        # '_thumb.png',
    ]

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    
    # nbody = 3
    # store_folder = store_folder+str(nbody)

    store_folder_a = os.path.join(store_folder,'copy_cst1')
    store_folder_b = os.path.join(store_folder,'copy_cst2')

    folder_list = [
        store_folder_a,
        store_folder_b,
    ]


    fuse_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/fuse/')

    # if os.path.isdir(copy_folder):
    #     shutil.rmtree(copy_folder)
    # 
    # os.makedirs(fuse_folder)


    hash_dict_list = []

    for folder in folder_list:

        hash_dict = {}
        choreo.SelectFiles_Action(folder,hash_dict)
        hash_dict_list.append(hash_dict)


    rtol = 1e-5
    detect_multiples = True
    only_Action = False

    n_master = 0
    master_hash_dict = {}
    for hash_dict, folder in zip(hash_dict_list,folder_list):

        for key_new, val_new in hash_dict.items():

            Already_in = False

            for key_master, val_master in master_hash_dict.items():

                IsCandidate = choreo.TestHashSame(val_new, val_master, rtol = rtol, detect_multiples = detect_multiples, only_Action = only_Action)

                Already_in = Already_in or IsCandidate
            
            if not(Already_in):

                n_master += 1

                key_master = str(n_master).zfill(5)

                master_hash_dict[key_master] = val_new

                for ext in ext_list:

                    source = os.path.join(folder, key_new + ext)
                    dest = os.path.join(fuse_folder, key_master + ext)
                    shutil.copy(source,dest)




if __name__ == "__main__":
    main()    
