'''
ODE.py : Defines ODE-related things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m
import mpmath
import functools

from choreo.scipy_plus.cython.SegmQuad import QuadFormula
from choreo.scipy_plus.SegmQuad        import QuadFrom3Term

from choreo.scipy_plus.cython.ODE import ExplicitSymplecticWithTable_XV_cython
from choreo.scipy_plus.cython.ODE import ExplicitSymplecticWithTable_VX_cython
from choreo.scipy_plus.cython.ODE import SymplecticStormerVerlet_XV_cython
from choreo.scipy_plus.cython.ODE import SymplecticStormerVerlet_VX_cython
from choreo.scipy_plus.cython.ODE import ImplicitSymplecticWithTableGaussSeidel_VX_cython
from choreo.scipy_plus.cython.ODE import ImplicitSymplecticTanWithTableGaussSeidel_VX_cython

from choreo.scipy_plus.cython.ODE import ImplicitSymplecticWithTableGaussSeidel_VX_cython_mulfun
from choreo.scipy_plus.cython.ODE import ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_mulfun

from choreo.scipy_plus.multiprec_tables import ComputeGaussButcherTables


#####################
# EXPLICIT RK STUFF #
#####################

c_table_Euler = np.array([1.])
d_table_Euler = np.array([1.])
assert c_table_Euler.size == d_table_Euler.size
nsteps_Euler = c_table_Euler.size
SymplecticEuler_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)
SymplecticEuler_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)

c_table_StormerVerlet = np.array([0.    ,1.      ])
d_table_StormerVerlet = np.array([1./2  ,1./2    ])
assert c_table_StormerVerlet.size == d_table_StormerVerlet.size
nsteps_StormerVerlet = c_table_StormerVerlet.size
SymplecticStormerVerlet_Table_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_StormerVerlet,d_table=d_table_StormerVerlet,nsteps=nsteps_StormerVerlet)
SymplecticStormerVerlet_Table_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_StormerVerlet,d_table=d_table_StormerVerlet,nsteps=nsteps_StormerVerlet)

c_table_Ruth3 = np.array([1.        ,-2./3  ,2/3    ])
d_table_Ruth3 = np.array([-1./24    , 3./4  ,7./24  ])
assert c_table_Ruth3.size == d_table_Ruth3.size
nsteps_Ruth3 = c_table_Ruth3.size
SymplecticRuth3_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)
SymplecticRuth3_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)

curt2 = m.pow(2,1./3)
c_table_Ruth4 = np.array([1./(2*(2-curt2))  ,(1-curt2)/(2*(2-curt2))    ,(1-curt2)/(2*(2-curt2))    ,1./(2*(2-curt2))   ])
d_table_Ruth4 = np.array([1./(2-curt2)      ,-curt2/(2-curt2)           ,1./(2-curt2)               ,0.                 ])
assert c_table_Ruth4.size == d_table_Ruth4.size
nsteps_Ruth4 = c_table_Ruth4.size
SymplecticRuth4_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)
SymplecticRuth4_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)

c_table_Ruth4Rat = np.array([0.     , 1./3  , -1./3     , 1.        , -1./3 , 1./3  ])
d_table_Ruth4Rat = np.array([7./48  , 3./8  , -1./48    , -1./48    ,  3./8 , 7./48 ])
assert c_table_Ruth4Rat.size == d_table_Ruth4Rat.size
nsteps_Ruth4Rat = c_table_Ruth4Rat.size
SymplecticRuth4Rat_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Ruth4Rat,d_table=d_table_Ruth4Rat,nsteps=nsteps_Ruth4Rat)
SymplecticRuth4Rat_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Ruth4Rat,d_table=d_table_Ruth4Rat,nsteps=nsteps_Ruth4Rat)


#####################
# IMPLICIT RK STUFF #
#####################


@functools.cache
def ComputeGaussButcherTables_np(n,dps=30):

    mpmath.mp.dps = dps
    Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = ComputeGaussButcherTables(n)

    Butcher_a_np = np.array(Butcher_a.tolist(),dtype=np.float64)
    Butcher_b_np = np.array(Butcher_b.tolist(),dtype=np.float64).reshape(n)
    Butcher_c_np = np.array(Butcher_c.tolist(),dtype=np.float64).reshape(n)
    Butcher_beta_np = np.array(Butcher_beta.tolist(),dtype=np.float64)
    Butcher_gamma_np = np.array(Butcher_gamma.tolist(),dtype=np.float64)

    return Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, Butcher_gamma_np


######################################################################


nsteps_LobattoIIIA_3 = 3
a_table_LobattoIIIA_3 = np.array( 
    [   [ 0     , 0     , 0     ],
        [ 5/24  , 1/3   , -1/24 ],
        [ 1/6   , 2/3   , 1/6   ]   ],
    dtype = np.float64)
b_table_LobattoIIIA_3 = np.array([ 1/6  , 2/3   , 1/6 ], dtype = np.float64)
c_table_LobattoIIIA_3 = np.array([ 0    , 1/2   , 1   ], dtype = np.float64)
beta_table_LobattoIIIA_3 = np.zeros((nsteps_LobattoIIIA_3,nsteps_LobattoIIIA_3), dtype = np.float64)
gamma_table_LobattoIIIA_3 = np.zeros((nsteps_LobattoIIIA_3,nsteps_LobattoIIIA_3), dtype = np.float64)

sqrt5 = m.sqrt(5)
nsteps_LobattoIIIA_4 = 4
a_table_LobattoIIIA_4 = np.array( 
    [   [ 0                 , 0                     , 0                 , 0                 ],
        [ (11+sqrt5)/120    , (25-sqrt5)/120        , (25-13*sqrt5)/120 , (-1+sqrt5)/120    ],
        [ (11-sqrt5)/120    , (25+13*sqrt5)/120     , (25+sqrt5)/120    , (-1-sqrt5)/120    ],
        [ 1/12              , 5/12                  , 5/12              , 1/12              ]   ],
    dtype = np.float64)
b_table_LobattoIIIA_4 = np.array([ 1/12 , 5/12          , 5/12          , 1/12 ], dtype = np.float64)
c_table_LobattoIIIA_4 = np.array([ 0    , (5-sqrt5)/10  , (5+sqrt5)/10  , 1    ], dtype = np.float64)
beta_table_LobattoIIIA_4 = np.zeros((nsteps_LobattoIIIA_3,nsteps_LobattoIIIA_4), dtype = np.float64)
gamma_table_LobattoIIIA_4 = np.zeros((nsteps_LobattoIIIA_3,nsteps_LobattoIIIA_4), dtype = np.float64)

#######################################################################################################


nsteps_LobattoIIIB_3 = 3
a_table_LobattoIIIB_3 = np.array( 
    [   [ 1/6   , -1/6  , 0 ],
        [ 1/6   , 1/3   , 0 ],
        [ 1/6   , 5/6   , 0 ]   ],
    dtype = np.float64)
b_table_LobattoIIIB_3 = np.array([ 1/6  , 2/3   , 1/6 ], dtype = np.float64)
c_table_LobattoIIIB_3 = np.array([ 0    , 1/2   , 1   ], dtype = np.float64)
beta_table_LobattoIIIB_3 = np.zeros((nsteps_LobattoIIIB_3,nsteps_LobattoIIIB_3), dtype = np.float64)
gamma_table_LobattoIIIB_3 = np.zeros((nsteps_LobattoIIIB_3,nsteps_LobattoIIIB_3), dtype = np.float64)

nsteps_LobattoIIIB_4 = 4
a_table_LobattoIIIB_4 = np.array( 
    [   [ 1/12  , (-1-sqrt5)/24     , (-1+sqrt5)/24     , 0 ],
        [ 1/12  , (25+sqrt5)/120    , (25-13*sqrt5)/120 , 0 ],
        [ 1/12  , (25+13*sqrt5)/120 , (25-sqrt5)/120    , 0 ],
        [ 1/12  , (11-sqrt5)/24     , (11+sqrt5)/24     , 0 ]   ],
    dtype = np.float64)
b_table_LobattoIIIB_4 = np.array([ 1/12 , 5/12          , 5/12          , 1/12 ], dtype = np.float64)
c_table_LobattoIIIB_4 = np.array([ 0    , (5-sqrt5)/10  , (5+sqrt5)/10  , 1    ], dtype = np.float64)
beta_table_LobattoIIIB_4 = np.zeros((nsteps_LobattoIIIB_4,nsteps_LobattoIIIB_4), dtype = np.float64)
gamma_table_LobattoIIIB_4 = np.zeros((nsteps_LobattoIIIB_4,nsteps_LobattoIIIB_4), dtype = np.float64)

#######################################################################

SymplecticLobattoIIIA_3 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_3,
    b_table_x = b_table_LobattoIIIA_3,
    c_table_x = c_table_LobattoIIIA_3,
    beta_table_x = beta_table_LobattoIIIA_3,
    a_table_v = a_table_LobattoIIIA_3,
    b_table_v = b_table_LobattoIIIA_3,
    c_table_v = c_table_LobattoIIIA_3,
    beta_table_v = beta_table_LobattoIIIA_3,
    nsteps = nsteps_LobattoIIIA_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticLobattoIIIB_3 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_3,
    b_table_x = b_table_LobattoIIIB_3,
    c_table_x = c_table_LobattoIIIB_3,
    beta_table_x = beta_table_LobattoIIIB_3,
    a_table_v = a_table_LobattoIIIB_3,
    b_table_v = b_table_LobattoIIIB_3,
    c_table_v = c_table_LobattoIIIB_3,
    beta_table_v = beta_table_LobattoIIIB_3,
    nsteps = nsteps_LobattoIIIB_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticLobattoIIIA_4 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_4,
    b_table_x = b_table_LobattoIIIA_4,
    c_table_x = c_table_LobattoIIIA_4,
    beta_table_x = beta_table_LobattoIIIA_4,
    a_table_v = a_table_LobattoIIIA_4,
    b_table_v = b_table_LobattoIIIA_4,
    c_table_v = c_table_LobattoIIIA_4,
    beta_table_v = beta_table_LobattoIIIA_4,
    nsteps = nsteps_LobattoIIIA_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticLobattoIIIB_4 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_4,
    b_table_x = b_table_LobattoIIIB_4,
    c_table_x = c_table_LobattoIIIB_4,
    beta_table_x = beta_table_LobattoIIIB_4,
    a_table_v = a_table_LobattoIIIB_4,
    b_table_v = b_table_LobattoIIIB_4,
    c_table_v = c_table_LobattoIIIB_4,
    beta_table_v = beta_table_LobattoIIIB_4,
    nsteps = nsteps_LobattoIIIB_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticPartitionedLobattoIII_AX_BV_3 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_3,
    b_table_x = b_table_LobattoIIIA_3,
    c_table_x = c_table_LobattoIIIA_3,
    beta_table_x = beta_table_LobattoIIIA_3,
    a_table_v = a_table_LobattoIIIB_3,
    b_table_v = b_table_LobattoIIIB_3,
    c_table_v = c_table_LobattoIIIB_3,
    beta_table_v = beta_table_LobattoIIIB_3,
    nsteps = nsteps_LobattoIIIB_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticPartitionedLobattoIII_AV_BX_3 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_3,
    b_table_x = b_table_LobattoIIIB_3,
    c_table_x = c_table_LobattoIIIB_3,
    beta_table_x = beta_table_LobattoIIIB_3,
    a_table_v = a_table_LobattoIIIA_3,
    b_table_v = b_table_LobattoIIIA_3,
    c_table_v = c_table_LobattoIIIA_3,
    beta_table_v = beta_table_LobattoIIIA_3,
    nsteps = nsteps_LobattoIIIB_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticPartitionedLobattoIII_AX_BV_4 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_4,
    b_table_x = b_table_LobattoIIIA_4,
    c_table_x = c_table_LobattoIIIA_4,
    beta_table_x = beta_table_LobattoIIIA_4,
    a_table_v = a_table_LobattoIIIB_4,
    b_table_v = b_table_LobattoIIIB_4,
    c_table_v = c_table_LobattoIIIB_4,
    beta_table_v = beta_table_LobattoIIIB_4,
    nsteps = nsteps_LobattoIIIB_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticPartitionedLobattoIII_AV_BX_4 = functools.partial(
    ImplicitSymplecticWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_4,
    b_table_x = b_table_LobattoIIIB_4,
    c_table_x = c_table_LobattoIIIB_4,
    beta_table_x = beta_table_LobattoIIIB_4,
    a_table_v = a_table_LobattoIIIA_4,
    b_table_v = b_table_LobattoIIIA_4,
    c_table_v = c_table_LobattoIIIA_4,
    beta_table_v = beta_table_LobattoIIIA_4,
    nsteps = nsteps_LobattoIIIB_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

##############################################################################

SymplecticTanLobattoIIIA_3 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_3,
    b_table_x = b_table_LobattoIIIA_3,
    c_table_x = c_table_LobattoIIIA_3,
    beta_table_x = beta_table_LobattoIIIA_3,
    a_table_v = a_table_LobattoIIIA_3,
    b_table_v = b_table_LobattoIIIA_3,
    c_table_v = c_table_LobattoIIIA_3,
    beta_table_v = beta_table_LobattoIIIA_3,
    nsteps = nsteps_LobattoIIIA_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticTanLobattoIIIB_3 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_3,
    b_table_x = b_table_LobattoIIIB_3,
    c_table_x = c_table_LobattoIIIB_3,
    beta_table_x = beta_table_LobattoIIIB_3,
    a_table_v = a_table_LobattoIIIB_3,
    b_table_v = b_table_LobattoIIIB_3,
    c_table_v = c_table_LobattoIIIB_3,
    beta_table_v = beta_table_LobattoIIIB_3,
    nsteps = nsteps_LobattoIIIB_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticTanLobattoIIIA_4 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_4,
    b_table_x = b_table_LobattoIIIA_4,
    c_table_x = c_table_LobattoIIIA_4,
    beta_table_x = beta_table_LobattoIIIA_4,
    a_table_v = a_table_LobattoIIIA_4,
    b_table_v = b_table_LobattoIIIA_4,
    c_table_v = c_table_LobattoIIIA_4,
    beta_table_v = beta_table_LobattoIIIA_4,
    nsteps = nsteps_LobattoIIIA_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticTanLobattoIIIB_4 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_4,
    b_table_x = b_table_LobattoIIIB_4,
    c_table_x = c_table_LobattoIIIB_4,
    beta_table_x = beta_table_LobattoIIIB_4,
    a_table_v = a_table_LobattoIIIB_4,
    b_table_v = b_table_LobattoIIIB_4,
    c_table_v = c_table_LobattoIIIB_4,
    beta_table_v = beta_table_LobattoIIIB_4,
    nsteps = nsteps_LobattoIIIB_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticTanPartitionedLobattoIII_AX_BV_3 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_3,
    b_table_x = b_table_LobattoIIIA_3,
    c_table_x = c_table_LobattoIIIA_3,
    beta_table_x = beta_table_LobattoIIIA_3,
    a_table_v = a_table_LobattoIIIB_3,
    b_table_v = b_table_LobattoIIIB_3,
    c_table_v = c_table_LobattoIIIB_3,
    beta_table_v = beta_table_LobattoIIIB_3,
    nsteps = nsteps_LobattoIIIB_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticTanPartitionedLobattoIII_AV_BX_3 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_3,
    b_table_x = b_table_LobattoIIIB_3,
    c_table_x = c_table_LobattoIIIB_3,
    beta_table_x = beta_table_LobattoIIIB_3,
    a_table_v = a_table_LobattoIIIA_3,
    b_table_v = b_table_LobattoIIIA_3,
    c_table_v = c_table_LobattoIIIA_3,
    beta_table_v = beta_table_LobattoIIIA_3,
    nsteps = nsteps_LobattoIIIB_3,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticTanPartitionedLobattoIII_AX_BV_4 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIA_4,
    b_table_x = b_table_LobattoIIIA_4,
    c_table_x = c_table_LobattoIIIA_4,
    beta_table_x = beta_table_LobattoIIIA_4,
    a_table_v = a_table_LobattoIIIB_4,
    b_table_v = b_table_LobattoIIIB_4,
    c_table_v = c_table_LobattoIIIB_4,
    beta_table_v = beta_table_LobattoIIIB_4,
    nsteps = nsteps_LobattoIIIB_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

SymplecticTanPartitionedLobattoIII_AV_BX_4 = functools.partial(
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
    a_table_x = a_table_LobattoIIIB_4,
    b_table_x = b_table_LobattoIIIB_4,
    c_table_x = c_table_LobattoIIIB_4,
    beta_table_x = beta_table_LobattoIIIB_4,
    a_table_v = a_table_LobattoIIIA_4,
    b_table_v = b_table_LobattoIIIA_4,
    c_table_v = c_table_LobattoIIIA_4,
    beta_table_v = beta_table_LobattoIIIA_4,
    nsteps = nsteps_LobattoIIIB_4,
    eps = np.finfo(np.float64).eps,
    maxiter = 50
)

##############################################################################


all_SymplecticIntegrators = {
    'SymplecticEuler'               : SymplecticEuler_XV,
    'SymplecticEuler_XV'            : SymplecticEuler_XV,
    'SymplecticEuler_VX'            : SymplecticEuler_VX,
    'SymplecticStormerVerlet'       : SymplecticStormerVerlet_XV_cython,
    # 'SymplecticStormerVerlet_Table_VX'    : SymplecticStormerVerlet_Table_VX,
    # 'SymplecticStormerVerlet_Table_XV'    : SymplecticStormerVerlet_Table_XV,
    'SymplecticStormerVerlet_XV'    : SymplecticStormerVerlet_XV_cython,
    'SymplecticStormerVerlet_VX'    : SymplecticStormerVerlet_VX_cython,
    'SymplecticRuth3'               : SymplecticRuth3_XV,
    'SymplecticRuth3_XV'            : SymplecticRuth3_XV,
    'SymplecticRuth3_VX'            : SymplecticRuth3_VX,
    'SymplecticRuth4'               : SymplecticRuth4_XV,
    'SymplecticRuth4_XV'            : SymplecticRuth4_XV,
    'SymplecticRuth4_VX'            : SymplecticRuth4_VX,
    'SymplecticRuth4Rat'            : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_XV'         : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_VX'         : SymplecticRuth4Rat_VX,
    'LobattoIIIA_3'                 : SymplecticLobattoIIIA_3,
    'LobattoIIIB_3'                 : SymplecticLobattoIIIB_3,
    'LobattoIIIA_4'                 : SymplecticLobattoIIIA_4,
    'LobattoIIIB_4'                 : SymplecticLobattoIIIB_4,
    'PartitionedLobattoIII_AX_BV_3' :SymplecticPartitionedLobattoIII_AX_BV_3,
    'PartitionedLobattoIII_AV_BX_3' :SymplecticPartitionedLobattoIII_AV_BX_3,
    'PartitionedLobattoIII_AX_BV_4' :SymplecticPartitionedLobattoIII_AX_BV_4,
    'PartitionedLobattoIII_AV_BX_4' :SymplecticPartitionedLobattoIII_AV_BX_4,
    }

all_SymplecticTanIntegrators = {
    'LobattoIIIA_3'                 : SymplecticTanLobattoIIIA_3,
    'LobattoIIIB_3'                 : SymplecticTanLobattoIIIB_3,
    'LobattoIIIA_4'                 : SymplecticTanLobattoIIIA_4,
    'LobattoIIIB_4'                 : SymplecticTanLobattoIIIB_4,
    'PartitionedLobattoIII_AX_BV_3' :SymplecticTanPartitionedLobattoIII_AX_BV_3,
    'PartitionedLobattoIII_AV_BX_3' :SymplecticTanPartitionedLobattoIII_AV_BX_3,
    'PartitionedLobattoIII_AX_BV_4' :SymplecticTanPartitionedLobattoIII_AX_BV_4,
    'PartitionedLobattoIII_AV_BX_4' :SymplecticTanPartitionedLobattoIII_AV_BX_4,
    }

all_unique_SymplecticIntegrators = {
    'SymplecticEuler_XV'            : SymplecticEuler_XV,
    'SymplecticEuler_VX'            : SymplecticEuler_VX,
    # 'SymplecticStormerVerlet_Table_VX'    : SymplecticStormerVerlet_Table_VX,
    'SymplecticStormerVerlet_XV'    : SymplecticStormerVerlet_XV_cython,
    # 'SymplecticStormerVerlet_Table_XV'    : SymplecticStormerVerlet_Table_XV,
    'SymplecticStormerVerlet_VX'    : SymplecticStormerVerlet_VX_cython,
    'SymplecticRuth3_XV'            : SymplecticRuth3_XV,
    'SymplecticRuth3_VX'            : SymplecticRuth3_VX,
    'SymplecticRuth4_XV'            : SymplecticRuth4_XV,
    'SymplecticRuth4_VX'            : SymplecticRuth4_VX,
    'SymplecticRuth4Rat_XV'         : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_VX'         : SymplecticRuth4Rat_VX,
    }

def GetSymplecticIntegrator(method='SymplecticRuth3', mul_x = True):

    integrator = all_SymplecticIntegrators.get(method)

    if integrator is None:

        if method.startswith("SymplecticGauss"):

            descr = method.removeprefix("SymplecticGauss")
            n = int(descr)
            Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, _ = ComputeGaussButcherTables_np(n)

            if mul_x:
                ImplicitSymplecticWithTableGaussSeidel = ImplicitSymplecticWithTableGaussSeidel_VX_cython_mulfun
            else:
                ImplicitSymplecticWithTableGaussSeidel = ImplicitSymplecticWithTableGaussSeidel_VX_cython

            integrator = functools.partial(
                ImplicitSymplecticWithTableGaussSeidel,
                a_table_x = Butcher_a_np,
                b_table_x = Butcher_b_np,
                c_table_x = Butcher_c_np,
                beta_table_x = Butcher_beta_np,
                a_table_v = Butcher_a_np,
                b_table_v = Butcher_b_np,
                c_table_v = Butcher_c_np,
                beta_table_v = Butcher_beta_np,
                nsteps = n,
                eps = np.finfo(np.float64).eps,
                maxiter = 50
            )

        else:
            raise ValueError(f"Method not found: {method}")

    return integrator

def GetSymplecticTanIntegrator(method='SymplecticGauss1', mul_x = True):

    integrator = all_SymplecticTanIntegrators.get(method)

    if integrator is None:

        if method.startswith("SymplecticGauss"):

            descr = method.removeprefix("SymplecticGauss")
            n = int(descr)
            Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, _ = ComputeGaussButcherTables_np(n)

            if mul_x:
                ImplicitSymplecticTanWithTableGaussSeidel = ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_mulfun
            else:
                ImplicitSymplecticTanWithTableGaussSeidel = ImplicitSymplecticTanWithTableGaussSeidel_VX_cython

            integrator = functools.partial(
                ImplicitSymplecticTanWithTableGaussSeidel,
                a_table_x = Butcher_a_np,
                b_table_x = Butcher_b_np,
                c_table_x = Butcher_c_np,
                beta_table_x = Butcher_beta_np,
                a_table_v = Butcher_a_np,
                b_table_v = Butcher_b_np,
                c_table_v = Butcher_c_np,
                beta_table_v = Butcher_beta_np,
                nsteps = n,
                eps = np.finfo(np.float64).eps,
                maxiter = 50
            )

        else:
            raise ValueError(f"Method not found: {method}")

    return integrator

