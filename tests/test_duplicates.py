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
    

    nbody = 3
    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)


    # Action_Hash_val = np.array([
    #     10.624096524001287,
    #     7.771489060335682,
    #     9.506013930674246,
    #     11.951765432717714,
    #     15.400375776702386
    # ])
    Action_Hash_val = np.array([
        13.207782336993958,
        8.940998633639229,
        11.423451535639789,
        15.505419300517392,
        22.332968571854142
    ])


    hash_dict = {}
    choreo.SelectFiles_Action(store_folder,hash_dict,Action_Hash_val=Action_Hash_val)





if __name__ == "__main__":
    main()    
