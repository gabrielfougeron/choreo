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

nsteps_list = list(range(1,5))
# nsteps_list = [3]
# dps_list = [15]
dps_list = [15,30,60,120,240]
# dps_list = [60]
overkill_dps = 1000



# # Compare overkill precision and working precision
# for nsteps in nsteps_list:
#     print('')
#     print(f'nsteps = {nsteps}')
# 
#     mpmath.mp.dps = overkill_dps
# 
#     Butcher_overkill = choreo.ComputeGaussButcherTables(nsteps)
# 
#     for dps in dps_list:
# 
#         mpmath.mp.dps = dps
#         
#         print('')
#         print(f'Current precision : {mpmath.mp.dps}')
# 
#         Butcher_regular = choreo.ComputeGaussButcherTables(nsteps)
# 
#         mpmath.mp.dps = overkill_dps
# 
#         for table_current, table_overkill in zip(Butcher_regular,Butcher_overkill):
# 
#             mpmath.nprint(mpmath.norm(table_current-table_overkill),print_dps)
#             # print(np.linalg.norm(np.array(table_current.tolist(),dtype=np.float64) - np.array(table_overkill.tolist(),dtype=np.float64)))



mpmath.mp.dps = overkill_dps

for nsteps in nsteps_list:
    print('')
    print(f'nsteps = {nsteps}')

    Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = choreo.ComputeGaussButcherTables(nsteps)
    # Butcher_a_ad, Butcher_b_ad, Butcher_c_ad = choreo.SymmetricAdjointButcher(Butcher_a, Butcher_b, Butcher_c, nsteps)
    Butcher_a_ad, Butcher_b_ad, Butcher_c_ad, Butcher_beta_ad, Butcher_gamma_ad = choreo.SymmetricAdjointButcher(Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma, nsteps)


    mpmath.nprint(mpmath.norm(Butcher_a   -Butcher_a_ad),print_dps)
    mpmath.nprint(mpmath.norm(Butcher_b   -Butcher_b_ad),print_dps)
    mpmath.nprint(mpmath.norm(Butcher_c   -Butcher_c_ad),print_dps)
    mpmath.nprint(mpmath.norm(Butcher_beta-Butcher_beta_ad),print_dps)
    mpmath.nprint(mpmath.norm(Butcher_gamma-Butcher_gamma_ad),print_dps)

