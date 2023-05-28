import os
import concurrent.futures
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# os.environ['NUMBA_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'


import json
import shutil
import random
import time
import math as m
import numpy as np
import scipy
import sys
import fractions
import functools

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','Simo_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','Simo_init_cond.txt')
all_NT_init = np.loadtxt(NT_init_filename)


keep_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','Simo_keep')


the_NT_init = range(len(all_NT_init))


all_ext = [
    '.json',
    '.npy',
    '.png',
]

for i in the_NT_init:

    basename = 'Simo_'+str(i).zfill(5)

    all_src_present = True
    all_dest_missing = True

    for ext in all_ext :

        src_filename = os.path.join(store_folder, basename+ext)
        all_src_present = all_src_present and os.path.isfile(src_filename)

        dest_filename = os.path.join(keep_folder, basename+ext)
        all_src_present = all_src_present and not(os.path.isfile(dest_filename))


    if all_src_present and all_dest_missing :

        for ext in all_ext :

            src_filename = os.path.join(store_folder, basename+ext)
            dest_filename = os.path.join(keep_folder, basename+ext)

            shutil.copyfile(src_filename, dest_filename)

    if not(all_src_present):

        print(f'Missing {basename}')

        
