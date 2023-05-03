'''
Choreo_scipy_plus.py : Define  ODE-related things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m

import mpmath
mpmath.mp.dps = 60 # Precision specified as a number of digits in the representation. Double precision = 15

import functools

from choreo.Choreo_cython_scipy_plus_ODE import ExplicitSymplecticWithTable_XV_cython
from choreo.Choreo_cython_scipy_plus_ODE import ExplicitSymplecticWithTable_VX_cython
from choreo.Choreo_cython_scipy_plus_ODE import SymplecticStormerVerlet_XV_cython
from choreo.Choreo_cython_scipy_plus_ODE import SymplecticStormerVerlet_VX_cython
from choreo.Choreo_cython_scipy_plus_ODE import ImplicitSymplecticWithTableGaussSeidel_VX_cython




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

# a_table_Gauss_1 = np.array([[1./2]])
# b_table_Gauss_1 = np.array([1.])
# c_table_Gauss_1 = np.array([1./2])
# nsteps_Gauss_1 = a_table_Gauss_1.shape[0]
# assert a_table_Gauss_1.shape[1] == nsteps_Gauss_1
# assert b_table_Gauss_1.size == nsteps_Gauss_1
# assert c_table_Gauss_1.size == nsteps_Gauss_1
# SymplecticGauss1_XV = functools.partial(ImplicitSymplecticWithTableGaussSeidel_VX_cython,a_table=a_table_Gauss_1,b_table=b_table_Gauss_1,c_table=c_table_Gauss_1,nsteps=nsteps_Gauss_1,eps=np.finfo(np.float64).eps,maxiter=1)
# 
# 
# db_G2 = m.pow(3,-1/2)/2
# a_table_Gauss_2 = np.array([[ 1./4  , 1./4 - db_G2  ],[ 1./4 + db_G2    , 1./4   ]])
# b_table_Gauss_2 = np.array([  1./2  , 1./2 ])
# c_table_Gauss_2 = np.array([  1./2 - db_G2 , 1./2 + db_G2 ])
# nsteps_Gauss_2 = a_table_Gauss_2.shape[0]
# assert a_table_Gauss_2.shape[1] == nsteps_Gauss_2
# assert b_table_Gauss_2.size == nsteps_Gauss_2
# assert c_table_Gauss_2.size == nsteps_Gauss_2
# SymplecticGauss2_XV = functools.partial(ImplicitSymplecticWithTableGaussSeidel_VX_cython,a_table=a_table_Gauss_2,b_table=b_table_Gauss_2,c_table=c_table_Gauss_2,nsteps=nsteps_Gauss_2,eps=np.finfo(np.float64).eps,maxiter=2)






def SafeGLIntOrder(N):

    n = m.ceil(N/2)

    if (((n%2) == 1) and ((N%2) == 1)) or ((n-1) == N):
        n += 1

    return n

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

def ComputeButcher_a(a,b,n,z=None,wint=None,zint=None,nint=None):

    if z is None:
        w, z = QuadFrom3Term(a,b,n)

    if (nint is None) or (wint is None) or (zint is None):
        assert (nint is None) and (wint is None) and (zint is None)
        
        nint = SafeGLIntOrder(n)
        aint, bint = ShiftedGaussLegendre3Term(nint)
        wint, zint = QuadFrom3Term(aint,bint,nint)

    Butcher_a = mpmath.matrix(n)

    for iint in range(nint):

        for i in range(n):

            lag = EvalLagrange(a,b,n,z,z[i]*zint[iint])

            for j in range(n):

                Butcher_a[i,j] = Butcher_a[i,j] + wint[iint] * lag[j]

    for i in range(n):
        for j in range(n):

            Butcher_a[i,j] = z[i] * Butcher_a[i,j]

    return Butcher_a


def ComputeGaussButcherTables(n):

    a, b = ShiftedGaussLegendre3Term(n)
    w, z = QuadFrom3Term(a,b,n)

    Butcher_a = ComputeButcher_a(a,b,n,z)

    Butcher_a_np = np.array(Butcher_a.tolist(),dtype=np.float64)
    Butcher_b_np = np.array(w.tolist(),dtype=np.float64).reshape(n)
    Butcher_c_np = np.array(z.tolist(),dtype=np.float64).reshape(n)

    return Butcher_a_np, Butcher_b_np, Butcher_c_np



#######################################################################



all_SymplecticIntegrators = {
    'SymplecticEuler'               : SymplecticEuler_XV,
    'SymplecticEuler_XV'            : SymplecticEuler_XV,
    'SymplecticEuler_VX'            : SymplecticEuler_VX,
    'SymplecticStormerVerlet'       : SymplecticStormerVerlet_XV_cython,
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
    # 'SymplecticGauss1'              : SymplecticGauss1_XV,
    # 'SymplecticGauss2'              : SymplecticGauss2_XV,
}

all_unique_SymplecticIntegrators = {
    'SymplecticEuler_XV'            : SymplecticEuler_XV,
    'SymplecticEuler_VX'            : SymplecticEuler_VX,
    'SymplecticStormerVerlet_XV'    : SymplecticStormerVerlet_XV_cython,
    'SymplecticStormerVerlet_VX'    : SymplecticStormerVerlet_VX_cython,
    'SymplecticRuth3_XV'            : SymplecticRuth3_XV,
    'SymplecticRuth3_VX'            : SymplecticRuth3_VX,
    'SymplecticRuth4_XV'            : SymplecticRuth4_XV,
    'SymplecticRuth4_VX'            : SymplecticRuth4_VX,
    'SymplecticRuth4Rat_XV'         : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_VX'         : SymplecticRuth4Rat_VX,
    # 'SymplecticGauss1'              : SymplecticGauss1_XV,
    # 'SymplecticGauss2'              : SymplecticGauss2_XV,
}

def GetSymplecticIntegrator(method='SymplecticRuth3'):

    integrator = all_SymplecticIntegrators.get(method)

    if integrator is None:

        if method.startswith("SymplecticGauss"):

            descr = method.removeprefix("SymplecticGauss")
            n = int(descr)
            Butcher_a_np, Butcher_b_np, Butcher_c_np = ComputeGaussButcherTables(n)

            integrator = functools.partial(
                ImplicitSymplecticWithTableGaussSeidel_VX_cython,
                a_table = Butcher_a_np,
                b_table = Butcher_b_np,
                c_table = Butcher_c_np,
                nsteps = n,
                eps = np.finfo(np.float64).eps,
                maxiter = n
            )

        else:
            raise ValueError(f"Method not found: {method}")


    return integrator

