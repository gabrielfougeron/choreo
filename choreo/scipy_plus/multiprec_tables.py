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

def GaussMatFrom3Term(a,b,n):
    
    J =  mpmath.matrix(n)
    
    for i in range(n):
        J[i,i] = a[i]

    for i in range(n-1):

        J[i  ,i+1] = mpmath.sqrt(b[i+1])
        J[i+1,i  ] = J[i  ,i+1]

    return J

def LobattoMatFrom3Term(a,b,n):
    
    J =  mpmath.matrix(n)
    
    for i in range(n-1):
        J[i,i] = a[i]

    for i in range(n-2):

        J[i  ,i+1] = mpmath.sqrt(b[i+1])
        J[i+1,i  ] = J[i  ,i+1]

    m = n-1

    xm = mpmath.mpf(0)
    phim = EvalAllFrom3Term(a,b,m,xm)
    xp = mpmath.mpf(1)
    phip = EvalAllFrom3Term(a,b,m,xp)

    mat = mpmath.matrix(2)
    
    mat[0,0] = phim[m]
    mat[1,0] = phip[m]
    mat[0,1] = phim[m-1]
    mat[1,1] = phip[m-1]
    
    rhs = mpmath.matrix(2,1)
    rhs[0] = xm * phim[m]
    rhs[1] = xp * phip[m]

    (alpha, beta) = (mat ** -1) * rhs

    J[n-1, n-1] = alpha
    J[n-2, n-1] = mpmath.sqrt(beta)
    J[n-1, n-2] = J[n-2, n-1]

    return J

def RadauMatFrom3Term(a,b,n,x):
    
    J =  mpmath.matrix(n)
    
    for i in range(n-1):
        J[i,i] = a[i]

    for i in range(n-1):

        J[i  ,i+1] = mpmath.sqrt(b[i+1])
        J[i+1,i  ] = J[i  ,i+1]

    m = n-1
    phim = EvalAllFrom3Term(a,b,m,x)
    alpha = x - b[m] * phim[m-1] / phim[m]

    J[n-1, n-1] = alpha

    return J

def QuadFrom3Term(a,b,n, method = "Gauss"):

    if method == "Gauss":
        J = GaussMatFrom3Term(a,b,n)
    elif method == "Radau_I":
        x = mpmath.mpf(0)
        J = RadauMatFrom3Term(a,b,n,x)
    elif method == "Radau_II":
        x = mpmath.mpf(1)
        J = RadauMatFrom3Term(a,b,n,x)
    elif method == "Lobatto_III":
        J = LobattoMatFrom3Term(a,b,n)
    else:
        raise ValueError(f"Unknown method {method}")
    
    z, P = mpmath.mp.eigsy(J)

    w = mpmath.matrix(n,1)
    for i in range(n):
        w[i] = b[0] * P[0,i] * P[0,i]

    return w, z

def BuildButcherCMat(z,n,m):
    
    mat = mpmath.matrix(n,m)

    for j in range(n):
        for i in range(m):
            mat[j,i] = z[j]**i
            
    return mat

def BuildButcherCRHS(y,z,n,m):
    
    rhs = mpmath.matrix(n,m)

    for j in range(n):
        for i in range(m):
            rhs[j,i] = (z[j]**(i+1) - y[j]**(i+1))/(i+1)
            
    return rhs

def ComputeButcher_collocation(z,n):

    mat = BuildButcherCMat(z,n,n)
    mat_inv = mat ** (-1)
    
    y = mpmath.matrix(n,1)
    for i in range(n):
        y[i] = 0
    
    rhs = BuildButcherCRHS(y,z,n,n)
    Butcher_a = rhs * mat_inv
    
    zp = mpmath.matrix(n,1)
    for i in range(n):
        y[i]  = 1
        zp[i] = 1+z[i]
        
    rhs = BuildButcherCRHS(y,zp,n,n)
    Butcher_beta = rhs * mat_inv    
    
    zp = mpmath.matrix(n,1)
    for i in range(n):
        y[i]  = -1+z[i]
        zp[i] = 0
        
    rhs = BuildButcherCRHS(y,zp,n,n)
    Butcher_gamma = rhs * mat_inv
    
    return Butcher_a, Butcher_beta, Butcher_gamma

def ComputeButcher_sub_collocation(z,n):

    mat = BuildButcherCMat(z,n-1,n-1)
    mat_inv = mat ** (-1)
    
    y = mpmath.matrix(n,1)
    for i in range(n):
        y[i] = 0
    
    rhs_plus = BuildButcherCRHS(y,z,n,n)
    rhs = rhs_plus[1:n,0:(n-1)]
    Butcher_a_sub = rhs * mat_inv
    
    Butcher_a = mpmath.matrix(n)
    for i in range(n-1):
        for j in range(n-1):
            Butcher_a[i+1,j] = Butcher_a_sub[i,j]
            
    return Butcher_a


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

def SymplecticAdjointButcher(Butcher_a, Butcher_b, n):

    Butcher_a_ad = mpmath.matrix(n)

    for i in range(n):
        for j in range(n):
            
            Butcher_a_ad[i,j] = Butcher_b[j] * (1 - Butcher_a[j,i] / Butcher_b[i])
            
    return Butcher_a_ad

@functools.cache
def ComputeGaussButcherTables(n, dps=60, method="Gauss"):

    mpmath.mp.dps = dps

    a, b = ShiftedGaussLegendre3Term(n)
    
    for quad_method in ["Gauss", "Radau_II", "Radau_I", "Lobatto_III"]:
        if quad_method in method:
            w, z = QuadFrom3Term(a,b,n, method=quad_method)
            break
    else:
        return ValueError(f"Unknown associated quadrature method {method}")
    
    Butcher_a, Butcher_beta , Butcher_gamma = ComputeButcher_collocation(z, n)
    
    if method in ["Lobatto_IIIC", "Lobatto_IIIC*"]:
        Butcher_a = ComputeButcher_sub_collocation(z,n)
    
    known_method = False
    if method in ["Gauss", "Lobatto_IIIA", "Radau_IIA", "Lobatto_IIIC*"]:
        # No transformation is required
        known_method = True
        
    if method in ["Lobatto_IIIB", "Radau_IA", "Lobatto_IIIC"]:
        known_method = True
        # Symplectic adjoint
        Butcher_a = SymplecticAdjointButcher(Butcher_a, w, n)  
                  
    if method in ["Radau_IB", "Radau_IIB", "Lobatto_IIIS", "Lobatto_IIID" ]:
        known_method = True
        # Symplectic adjoint average
        Butcher_a_ad = SymplecticAdjointButcher(Butcher_a, w, n)    
        Butcher_a = (Butcher_a_ad + Butcher_a) / 2
        
    return Butcher_a, w, z, Butcher_beta, Butcher_gamma

def GetConvergenceRate(method, n):
    
    if "Gauss" in method:
        if n < 1:
            raise ValueError(f"Incorrect value for n {n}")
        th_cvg_rate = 2*n
    elif "Radau" in method:
        if n < 2:
            raise ValueError(f"Incorrect value for n {n}")
        th_cvg_rate = 2*n-1
    elif "Lobatto" in method:
        if n < 2:
            raise ValueError(f"Incorrect value for n {n}")
        th_cvg_rate = 2*n-2
    else:
        raise ValueError(f"Unknown method {method}")
    
    return th_cvg_rate

def ComputeImplicitSymplecticRKTable_Gauss(n, dps=60, method="Gauss"):

    th_cvg_rate = GetConvergenceRate(method, n)
    
    Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = ComputeGaussButcherTables(n, dps=dps, method=method)

    Butcher_a_np = np.array(Butcher_a.tolist(),dtype=np.float64)
    Butcher_b_np = np.array(Butcher_b.tolist(),dtype=np.float64).reshape(n)
    Butcher_c_np = np.array(Butcher_c.tolist(),dtype=np.float64).reshape(n)
    Butcher_beta_np = np.array(Butcher_beta.tolist(),dtype=np.float64)
    Butcher_gamma_np = np.array(Butcher_gamma.tolist(),dtype=np.float64)
    
    return ImplicitSymplecticRKTable(
        a_table     = Butcher_a_np      ,
        b_table     = Butcher_b_np      ,
        c_table     = Butcher_c_np      ,
        beta_table  = Butcher_beta_np   ,
        gamma_table = Butcher_gamma_np  ,
        th_cvg_rate = th_cvg_rate       ,
    )
        
def ComputeImplicitSymplecticRKTablePair_Gauss(n, dps=60, method="Gauss"):
    
    th_cvg_rate = GetConvergenceRate(method, n)
    
    Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = ComputeGaussButcherTables(n, dps=dps, method=method)
    Butcher_a_ad = SymplecticAdjointButcher(Butcher_a, Butcher_b, n)  
    
    Butcher_a_np = np.array(Butcher_a.tolist(),dtype=np.float64)
    Butcher_b_np = np.array(Butcher_b.tolist(),dtype=np.float64).reshape(n)
    Butcher_c_np = np.array(Butcher_c.tolist(),dtype=np.float64).reshape(n)
    Butcher_beta_np = np.array(Butcher_beta.tolist(),dtype=np.float64)
    Butcher_gamma_np = np.array(Butcher_gamma.tolist(),dtype=np.float64)
    Butcher_a_ad_np = np.array(Butcher_a_ad.tolist(),dtype=np.float64)
    
    rk = ImplicitSymplecticRKTable(
        a_table     = Butcher_a_np      ,
        b_table     = Butcher_b_np      ,
        c_table     = Butcher_c_np      ,
        beta_table  = Butcher_beta_np   ,
        gamma_table = Butcher_gamma_np  ,
        th_cvg_rate = th_cvg_rate       ,
    )
    
    rk_ad = ImplicitSymplecticRKTable(
        a_table     = Butcher_a_ad_np   ,
        b_table     = Butcher_b_np      ,
        c_table     = Butcher_c_np      ,
        beta_table  = Butcher_beta_np   ,
        gamma_table = Butcher_gamma_np  ,
        th_cvg_rate = th_cvg_rate       ,
    )
    
    return rk, rk_ad
        
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
    Variation on Yosida's method
    
    input : vector w as in Construction of higher order symplectic integrators in PHYSICS LETTERS A by Haruo Yoshida 1990.
    
    w[1:m+1] (m elements) is provided.

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
    
    