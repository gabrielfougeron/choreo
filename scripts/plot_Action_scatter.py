import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import concurrent.futures
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
import matplotlib.pyplot as plt


def main():

    cmap = 'Blues'

    nbody = 5

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    hash_dict = {}
    _ = choreo.SelectFiles_Action(store_folder,hash_dict)

    n_sols = len(hash_dict)

    x = np.zeros(n_sols)
    y = np.zeros(n_sols)
    c = np.zeros(n_sols)

    for key, value in hash_dict.items():
        
        isol = int(key)-1

#         x[isol] = value[0]
#         y[isol] = value[1][0]
# 
        x[isol] = value[1][0]
        y[isol] = value[1][2]



        c[isol] = isol

    plt.scatter(x, y, c=c, cmap=cmap)
    plt.savefig(os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/plot_Action_scatter.png'))


if __name__ == "__main__":
    main()    
