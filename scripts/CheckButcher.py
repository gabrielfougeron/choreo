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


# 
# 
# for nsteps in nsteps_list:
#     print('')
#     print(f'nsteps = {nsteps}')
#     
#     mpmath.mp.dps = 100
# 
#     tbeg = time.perf_counter()
#     Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = choreo.scipy_plus.multiprec_tables.ComputeGaussButcherTables(nsteps)
#     tend = time.perf_counter()
#     print(f"Time old = {tend-tbeg}")
#     
#     
#     mpmath.mp.dps = 100
#     
#     tbeg = time.perf_counter()
#     a, b = choreo.scipy_plus.multiprec_tables.ShiftedGaussLegendre3Term(nsteps)
#     w, z = choreo.scipy_plus.multiprec_tables.QuadFrom3Term(a,b,nsteps)
#     Butcher_a_new, Butcher_beta_new , Butcher_gamma_new = choreo.scipy_plus.multiprec_tables.ComputeButcher_collocation(z,nsteps)
#     tend = time.perf_counter()
#     print(f"Time new = {tend-tbeg}")
#     
#     print('Error')
#     mpmath.nprint(mpmath.norm(Butcher_a - Butcher_a_new),print_dps)
#     mpmath.nprint(mpmath.norm(Butcher_beta - Butcher_beta_new),print_dps)
#     mpmath.nprint(mpmath.norm(Butcher_gamma - Butcher_gamma_new),print_dps)
#     

a_table_LobattoIIIA_3 = np.array( 
    [   [ 0     , 0     , 0     ],
        [ 5/24  , 1/3   , -1/24 ],
        [ 1/6   , 2/3   , 1/6   ]   ],
    dtype = np.float64)
b_table_LobattoIIIA_3 = np.array([ 1/6  , 2/3   , 1/6 ], dtype = np.float64)
c_table_LobattoIIIA_3 = np.array([ 0    , 1/2   , 1   ], dtype = np.float64)

sqrt5 = m.sqrt(5)
a_table_LobattoIIIA_4 = np.array( 
    [   [ 0                 , 0                     , 0                 , 0                 ],
        [ (11+sqrt5)/120    , (25-sqrt5)/120        , (25-13*sqrt5)/120 , (-1+sqrt5)/120    ],
        [ (11-sqrt5)/120    , (25+13*sqrt5)/120     , (25+sqrt5)/120    , (-1-sqrt5)/120    ],
        [ 1/12              , 5/12                  , 5/12              , 1/12              ]   ],
    dtype = np.float64)
b_table_LobattoIIIA_4 = np.array([ 1/12 , 5/12          , 5/12          , 1/12 ], dtype = np.float64)
c_table_LobattoIIIA_4 = np.array([ 0    , (5-sqrt5)/10  , (5+sqrt5)/10  , 1    ], dtype = np.float64)

a_table_LobattoIIIB_3 = np.array( 
    [   [ 1/6   , -1/6  , 0 ],
        [ 1/6   , 1/3   , 0 ],
        [ 1/6   , 5/6   , 0 ]   ],
    dtype = np.float64)
b_table_LobattoIIIB_3 = np.array([ 1/6  , 2/3   , 1/6 ], dtype = np.float64)
c_table_LobattoIIIB_3 = np.array([ 0    , 1/2   , 1   ], dtype = np.float64)

a_table_LobattoIIIB_4 = np.array( 
    [   [ 1/12  , (-1-sqrt5)/24     , (-1+sqrt5)/24     , 0 ],
        [ 1/12  , (25+sqrt5)/120    , (25-13*sqrt5)/120 , 0 ],
        [ 1/12  , (25+13*sqrt5)/120 , (25-sqrt5)/120    , 0 ],
        [ 1/12  , (11-sqrt5)/24     , (11+sqrt5)/24     , 0 ]   ],
    dtype = np.float64)
b_table_LobattoIIIB_4 = np.array([ 1/12 , 5/12          , 5/12          , 1/12 ], dtype = np.float64)
c_table_LobattoIIIB_4 = np.array([ 0    , (5-sqrt5)/10  , (5+sqrt5)/10  , 1    ], dtype = np.float64)


rk = choreo.scipy_plus.multiprec_tables.ComputeImplicitSymplecticRKTable_Gauss(4, dps=60, method="Lobatto")
rk_ad = rk.symplectic_adjoint()


# test_precomp = [a_table_LobattoIIIA_3, a_table_LobattoIIIB_3]
test_precomp = [a_table_LobattoIIIA_4, a_table_LobattoIIIB_4]
test_comp = [rk.a_table, rk_ad.a_table]

# print(a_table_LobattoIIIB_4)
# print(rk.a_table)
# print(rk_ad.a_table)
# 
# print(rk.a_table - rk_ad.a_table)

# 
for a,b in zip(test_precomp,test_comp):
    print(np.linalg.norm(b-a))




