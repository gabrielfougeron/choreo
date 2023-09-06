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

nsteps_list = list(range(2,10))
# nsteps_list = [2,3,4,5]
# nsteps_list = [2,3]
# dps_list = [15]
# dps_list = [15,30,60,120,240]
# dps_list = [60]
# overkill_dps = 1000

method_list = [
    "Gauss" ,
    "Radau_IA" ,
    "Radau_IIA" ,
    "Radau_IB" ,
    "Radau_IIB" ,
    "Lobatto_IIIA" ,
    "Lobatto_IIIB" ,
    "Lobatto_IIIC" ,
    "Lobatto_IIIC*" ,
    'Lobatto_IIID'  ,            
    'Lobatto_IIIS'  ,  
]


for nsteps in nsteps_list:
    print('')
    print('='*80)
    print('')
    print(f'nsteps = {nsteps}')

    for method in method_list:
        print('')
        # print('*'*60)
        # print('')
        print(f'{method = }')

        rk = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, dps=600, method=method)
        rk_ad = rk.symplectic_adjoint()

        print(rk.stability_cst)
        print(choreo.scipy_plus.ODE.ImplicitRKTable.symplectic_default(rk))
        print(rk.is_symplectic())


