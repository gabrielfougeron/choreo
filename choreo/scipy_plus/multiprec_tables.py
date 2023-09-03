'''
ODE.py : Defines ODE-related things I designed I feel ought to be in scipy.

'''

import functools
import math
import mpmath
import numpy as np

from choreo.scipy_plus.cython.SegmQuad import QuadFormula
from choreo.scipy_plus.cython.ODE import ExplicitSymplecticRKTable
from choreo.scipy_plus.cython.ODE import ImplicitSymplecticRKTable

def SafeGLIntOrder(N):

    n = math.ceil(N/2)

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
def ComputeGaussButcherTables_np(n,dps=30):

    mpmath.mp.dps = dps
    Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = ComputeGaussButcherTables(n)

    Butcher_a_np = np.array(Butcher_a.tolist(),dtype=np.float64)
    Butcher_b_np = np.array(Butcher_b.tolist(),dtype=np.float64).reshape(n)
    Butcher_c_np = np.array(Butcher_c.tolist(),dtype=np.float64).reshape(n)
    Butcher_beta_np = np.array(Butcher_beta.tolist(),dtype=np.float64)
    Butcher_gamma_np = np.array(Butcher_gamma.tolist(),dtype=np.float64)

    return Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, Butcher_gamma_np

def ComputeImplicitSymplecticRKTable_Gauss(n,dps=30):
    
    Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, Butcher_gamma_np = ComputeGaussButcherTables_np(n,dps=dps)
    
    return ImplicitSymplecticRKTable(
        a_table     = Butcher_a_np      ,
        b_table     = Butcher_b_np      ,
        c_table     = Butcher_c_np      ,
        beta_table  = Butcher_beta_np   ,
        gamma_table = Butcher_gamma_np  ,
        th_cvg_rate = 2*n               ,
    )
        
def Yoshida_w_to_cd(w_in, th_cvg_rate):
    '''
    input : vector w as in Construction of higher order symplectic integrators in PHYSICS LETTERS A by Haruo Yoshida 1990.
    
    w[1:m+1] (m elements) is provided. w0 is implicit.

    '''
    
    m = w_in.shape[0]
    
    wo = 1-2*math.fsum(w_in)
    w = np.zeros((m+1),dtype=np.float64)
    w[0] = wo
    for i in range(m):
        w[i+1] = w_in[i]
    
    n = 2*m + 2

    c_table = np.zeros((n),dtype=np.float64)    
    d_table = np.zeros((n),dtype=np.float64)   
    
    for i in range(m): 
        val = w[m-i]
        d_table[i]      = val
        d_table[2*m-i]  = val
    d_table[m] = w[0]
        
    c_table[0]     = w[m] / 2
    c_table[2*m+1] = w[m] / 2
    for i in range(m): 
        val = (w[m-i]+w[m-1-i]) / 2
        c_table[i+1]      = val
        c_table[2*m-i]  = val
        
    return ExplicitSymplecticRKTable(
        c_table     ,
        d_table     ,
        th_cvg_rate ,
    )
    
            
def Yoshida_w_to_cd_reduced(w, th_cvg_rate):
    '''
    input : vector w as in Construction of higher order symplectic integrators in PHYSICS LETTERS A by Haruo Yoshida 1990.
    
    w[1:m+1] (m elements) is provided. w0 is implicit.

    '''
    
    m = w.shape[0]
    n = 2*m

    c_table = np.zeros((n),dtype=np.float64)    
    d_table = np.zeros((n),dtype=np.float64)   
    
    for i in range(m): 
        val = w[m-1-i]
        d_table[i]      = val
        d_table[n-2-i]  = val

        
    c_table[0]   = w[m-1] / 2
    c_table[n-1] = w[m-1] / 2
    for i in range(m-1): 
        val = (w[m-1-i]+w[m-2-i]) / 2
        c_table[i+1]    = val
        c_table[n-2-i]  = val
        
    return ExplicitSymplecticRKTable(
        c_table     ,
        d_table     ,
        th_cvg_rate ,
    )
    
    