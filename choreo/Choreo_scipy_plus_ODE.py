'''
Choreo_scipy_plus.py : Define  ODE-related things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m

import mpmath

import functools

from choreo.Choreo_cython_scipy_plus_ODE import ExplicitSymplecticWithTable_XV_cython
from choreo.Choreo_cython_scipy_plus_ODE import ExplicitSymplecticWithTable_VX_cython
# from choreo.Choreo_cython_scipy_plus_ODE import SymplecticStormerVerlet_XV_cython
# from choreo.Choreo_cython_scipy_plus_ODE import SymplecticStormerVerlet_VX_cython
from choreo.Choreo_cython_scipy_plus_ODE import ImplicitSymplecticWithTableGaussSeidel_VX_cython
from choreo.Choreo_cython_scipy_plus_ODE import ImplicitSymplecticTanWithTableGaussSeidel_VX_cython
from choreo.Choreo_cython_scipy_plus_ODE import IntegrateOnSegment

from choreo.Choreo_cython_scipy_plus_ODE import ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas
from choreo.Choreo_cython_scipy_plus_ODE import ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas



#####################
# EXPLICIT RK STUFF #
#####################

c_table_Euler = np.array([1.])
d_table_Euler = np.array([1.])
assert c_table_Euler.size == d_table_Euler.size
nsteps_Euler = c_table_Euler.size
SymplecticEuler_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)
SymplecticEuler_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)

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


def SafeGLIntOrder(N):

    n = m.ceil(N/2)

    if (((n%2) == 1) and ((N%2) == 1)) or ((n-1) == N):
        n += 1

    return n

# 3 terms definition of polynomial families
# P_n+1 = (X - a_n) P_n - b_n P_n-1
def GaussLegendre3Term(n):

    a = mpmath.matrix(n,1)
    b = mpmath.matrix(n,1)

    b[0] = 2

    for i in range(1,n):

        i2 = i*i
        b[i] = mpmath.fraction(i2,4*i2-1)

    return a, b

def ShiftedGaussLegendre3Term(n):

    a = mpmath.matrix(n,1)
    b = mpmath.matrix(n,1)

    for i in range(n):
        a[i] = mpmath.fraction(1,2)

    b[0] = 1

    for i in range(1,n):

        i2 = i*i
        b[i] = mpmath.fraction(i2,4*(4*i2-1))

    return a, b

def EvalAllFrom3Term(a,b,n,x):
    # n >= 1

    phi = mpmath.matrix(n+1,1)

    phi[0] = mpmath.mpf(1)
    phi[1] = x - a[0]

    for i in range(1,n):

        phi[i+1] = (x - a[i]) * phi[i] - b[i] * phi[i-1]

    return phi

def EvalAllDerivFrom3Term(a,b,n,x,phi=None):
    # n >= 1

    if phi is None:
        phi = EvalAllFrom3Term(a,b,n,x)

    phip = mpmath.matrix(n+1,1)

    phip[0] = 0
    phip[1] = 1

    for i in range(1,n):

        phip[i+1] = (x - a[i]) * phip[i] - b[i] * phip[i-1] + phi[i]

    return phip

def MatFrom3Term(a,b,n):
    
    J =  mpmath.matrix(n)
    
    for i in range(n):
        J[i,i] = a[i]

    for i in range(n-1):

        J[i  ,i+1] = mpmath.sqrt(b[i+1])
        J[i+1,i  ] = J[i  ,i+1]

    return J

def QuadFrom3Term(a,b,n):

    J = MatFrom3Term(a,b,n)
    z, P = mpmath.mp.eigsy(J)

    w = mpmath.matrix(n,1)
    for i in range(n):
        w[i] = b[0] * P[0,i] * P[0,i]

    return w, z

def GatherDerivAtZeros(a,b,n,z=None):

    if z is None:
        w, z = QuadFrom3Term(a,b,n)

    phipz = mpmath.matrix(n,1)

    for i in range(n):
        
        phip = EvalAllDerivFrom3Term(a,b,n,z[i])
        phipz[i] = phip[n]

    return phipz

def EvalLagrange(a,b,n,z,x,phipz=None):

    if phipz is None :
          phipz = GatherDerivAtZeros(a,b,n,z)
    
    phi = EvalAllFrom3Term(a,b,n,x)
    
    lag = mpmath.matrix(n,1)
    
    for i in range(n):
    
        lag[i] = phi[n] / (phipz[i] * (x - z[i]))

    return lag

def ComputeButcher_psi(x,y,a,b,n,w=None,z=None,wint=None,zint=None,nint=None):

    if (w is None) or (z is None) :
        assert (w is None) and (z is None)
        w, z = QuadFrom3Term(a,b,n)

    if (nint is None) or (wint is None) or (zint is None):
        assert (nint is None) and (wint is None) and (zint is None)
        
        nint = SafeGLIntOrder(n)
        aint, bint = ShiftedGaussLegendre3Term(nint)
        wint, zint = QuadFrom3Term(aint,bint,nint)

    Butcher_psi = mpmath.matrix(n)

    for iint in range(nint):

        for i in range(n):

            tint = x[i] + (y[i]-x[i]) * zint[iint]

            lag = EvalLagrange(a,b,n,z,tint)

            for j in range(n):

                Butcher_psi[i,j] = Butcher_psi[i,j] + wint[iint] * lag[j]

    for i in range(n):
        for j in range(n):

            Butcher_psi[i,j] = (y[i]-x[i]) * Butcher_psi[i,j]

    return Butcher_psi

def ComputeButcher_a(a,b,n,w=None,z=None,wint=None,zint=None,nint=None):

    if (w is None) or (z is None) :
        assert (w is None) and (z is None)
        w, z = QuadFrom3Term(a,b,n)

    x = mpmath.matrix(n,1)
    return ComputeButcher_psi(x,z,a,b,n,w,z,wint,zint,nint)

def ComputeButcher_beta_gamma(a,b,n,w=None,z=None,wint=None,zint=None,nint=None):

    if (w is None) or (z is None) :
        w, z = QuadFrom3Term(a,b,n)

    x = mpmath.matrix(n,1)
    y = mpmath.matrix(n,1)

    for i in range(n):
        x[i] = 1
        y[i] = 1 + z[i]

    Butcher_beta = ComputeButcher_psi(x,y,a,b,n,w,z,wint,zint,nint)

    for i in range(n):
        x[i] = -1 + z[i]
        y[i] = 0

    Butcher_gamma = ComputeButcher_psi(x,y,a,b,n,w,z,wint,zint,nint)

    return Butcher_beta, Butcher_gamma

def ComputeGaussButcherTables(n):

    a, b = ShiftedGaussLegendre3Term(n)
    w, z = QuadFrom3Term(a,b,n)

    nint = SafeGLIntOrder(n)
    aint, bint = ShiftedGaussLegendre3Term(nint)
    wint, zint = QuadFrom3Term(aint,bint,nint)

    Butcher_a = ComputeButcher_a(a,b,n,w,z,wint,zint,nint)
    Butcher_beta, Butcher_gamma = ComputeButcher_beta_gamma(a,b,n,w,z,wint,zint,nint)

    return Butcher_a, w, z, Butcher_beta, Butcher_gamma

def SymmetricAdjointQuadrature(w,z,n):

    w_ad = mpmath.matrix(n,1)
    z_ad = mpmath.matrix(n,1)

    for i in range(n):

        z_ad[i] = 1 - z[n-1-i]
        w_ad[i] = w[n-1-i]

    return w_ad, z_ad

def SymmetricAdjointButcher(Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma, n):

    Butcher_b_ad, Butcher_c_ad = SymmetricAdjointQuadrature(Butcher_b,Butcher_c,n)

    Butcher_a_ad = mpmath.matrix(n)
    Butcher_beta_ad = mpmath.matrix(n)
    Butcher_gamma_ad = mpmath.matrix(n)

    for i in range(n):
        for j in range(n):
            
            Butcher_a_ad[i,j] = Butcher_b[n-1-j] - Butcher_a[n-1-i,n-1-j]

            Butcher_beta_ad[i,j]  = Butcher_gamma[n-1-i,n-1-j]
            Butcher_gamma_ad[i,j] = Butcher_beta[n-1-i,n-1-j]

    return Butcher_a_ad, Butcher_b_ad, Butcher_c_ad, Butcher_beta_ad, Butcher_gamma_ad

@functools.cache
def ComputeQuadrature_np(method,n,dps=30):

    if method == "Gauss" :
        a, b = ShiftedGaussLegendre3Term(n)
        
    else:
        raise ValueError(f"Method not found: {method}")
    
    w, z = QuadFrom3Term(a,b,n)

    w_np = np.array(w.tolist(),dtype=np.float64).reshape(n)
    z_np = np.array(z.tolist(),dtype=np.float64).reshape(n)
    
    return w_np, z_np

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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
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
    # 'SymplecticStormerVerlet'       : SymplecticStormerVerlet_XV_cython,
    # 'SymplecticStormerVerlet_XV'    : SymplecticStormerVerlet_XV_cython,
    # 'SymplecticStormerVerlet_VX'    : SymplecticStormerVerlet_VX_cython,
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
    # 'SymplecticStormerVerlet_XV'    : SymplecticStormerVerlet_XV_cython,
    # 'SymplecticStormerVerlet_VX'    : SymplecticStormerVerlet_VX_cython,
    'SymplecticRuth3_XV'            : SymplecticRuth3_XV,
    'SymplecticRuth3_VX'            : SymplecticRuth3_VX,
    'SymplecticRuth4_XV'            : SymplecticRuth4_XV,
    'SymplecticRuth4_VX'            : SymplecticRuth4_VX,
    'SymplecticRuth4Rat_XV'         : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_VX'         : SymplecticRuth4Rat_VX,
    }

def GetSymplecticIntegrator(method='SymplecticRuth3'):

    integrator = all_SymplecticIntegrators.get(method)

    if integrator is None:

        if method.startswith("SymplecticGauss"):

            descr = method.removeprefix("SymplecticGauss")
            n = int(descr)
            Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, _ = ComputeGaussButcherTables_np(n)

            integrator = functools.partial(
                ImplicitSymplecticWithTableGaussSeidel_VX_cython_blas,
                # ImplicitSymplecticWithTableGaussSeidel_VX_cython,
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

def GetSymplecticTanIntegrator(method='SymplecticGauss1'):


    integrator = all_SymplecticTanIntegrators.get(method)

    if integrator is None:

        if method.startswith("SymplecticGauss"):

            descr = method.removeprefix("SymplecticGauss")
            n = int(descr)
            Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, _ = ComputeGaussButcherTables_np(n)

            integrator = functools.partial(
                ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_blas,
                # ImplicitSymplecticTanWithTableGaussSeidel_VX_cython,
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

