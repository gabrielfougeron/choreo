import mpmath
import numpy as np


import os
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import math as m
import numpy as np
import sys
import time

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


print(f'Precision on load : {mpmath.mp.dps}')

print_dps = 15

nsteps_list = list(range(1,11))
# nsteps_list = list(range(1,3))
# dps_list = [15]
dps_list = [15,30,60,120,240]
overkill_dps = 1000


for nsteps in nsteps_list:
    print('')
    print(f'nsteps = {nsteps}')

    mpmath.mp.dps = overkill_dps

    Butcher_overkill = choreo.ComputeGaussButcherTables(nsteps)

    for dps in dps_list:

        mpmath.mp.dps = dps
        
        print('')
        print(f'Current precision : {mpmath.mp.dps}')

        Butcher_regular = choreo.ComputeGaussButcherTables(nsteps)

        mpmath.mp.dps = overkill_dps

        for table_current, table_overkill in zip(Butcher_regular,Butcher_overkill):

            mpmath.nprint(mpmath.norm(table_current-table_overkill),print_dps)
            # print(np.linalg.norm(np.array(table_current.tolist(),dtype=np.float64) - np.array(table_overkill.tolist(),dtype=np.float64)))

# 
# 
# for nsteps in nsteps_list:
#     print('')
#     print(f'nsteps = {nsteps}')
# 
#     Butcher_overkill = choreo.ComputeGaussButcherTables_np(nsteps,dps=overkill_dps)
# 
#     for dps in dps_list:
#         
#         print('')
#         print(f'Current precision : {dps}')
# 
#         Butcher_regular = choreo.ComputeGaussButcherTables_np(nsteps,dps=dps)
# 
#         for table_current, table_overkill in zip(Butcher_regular,Butcher_overkill):
# 
#             print(np.linalg.norm(table_current-table_overkill))

