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


# print("LU_decomp" in dir(mpmath))
# 
# for it in dir(mpmath):
#     
#     if 'solve' in it.lower():
#         
#         print(it)
# 
# exit()

print(f'Precision on load : {mpmath.mp.dps}')

print_dps = 15

nsteps_list = list(range(1,20))
# nsteps_list = [2]
# nsteps_list = [3]
# dps_list = [15]
# dps_list = [15,30,60,120,240]
# dps_list = [60]
# overkill_dps = 1000




for nsteps in nsteps_list:
    print('')
    print(f'nsteps = {nsteps}')
    
    mpmath.mp.dps = 100

    tbeg = time.perf_counter()
    Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = choreo.scipy_plus.multiprec_tables.ComputeGaussButcherTables(nsteps)
    tend = time.perf_counter()
    print(f"Time old = {tend-tbeg}")
    
    
    mpmath.mp.dps = 100
    
    tbeg = time.perf_counter()
    a, b = choreo.scipy_plus.multiprec_tables.ShiftedGaussLegendre3Term(nsteps)
    w, z = choreo.scipy_plus.multiprec_tables.QuadFrom3Term(a,b,nsteps)
    Butcher_a_new, Butcher_beta_new , Butcher_gamma_new = choreo.scipy_plus.multiprec_tables.ComputeButcher_collocation(z,nsteps)
    tend = time.perf_counter()
    print(f"Time new = {tend-tbeg}")
    
    print('Error')
    mpmath.nprint(mpmath.norm(Butcher_a - Butcher_a_new),print_dps)
    mpmath.nprint(mpmath.norm(Butcher_beta - Butcher_beta_new),print_dps)
    mpmath.nprint(mpmath.norm(Butcher_gamma - Butcher_gamma_new),print_dps)
    
