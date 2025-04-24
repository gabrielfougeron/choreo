'''
multiprec_tables.py : Computation of quadrature and Runge-Kutta tables in multiprecision.

'''

import functools
import math
import mpmath
import numpy as np

from choreo.segm.cython.quad    import QuadTable
from choreo.segm.cython.ODE     import ExplicitSymplecticRKTable
from choreo.segm.cython.ODE     import ImplicitRKTable

# Order is important.
all_GL_int = [
    "Gauss"         ,
    "Radau_II"      ,
    "Radau_I"       ,
    "Lobatto_III"   ,
]

def tridiag_eigenvalues(d, e):
    """
    Adapted from mpmath 
    """

    n = len(d)
    e[n-1] = 0
    iterlim = 2 * mpmath.mp.dps

    for l in range(n):
        j = 0
        while 1:
            m = l
            while 1:
                # look for a small subdiagonal element
                if m + 1 == n:
                    break
                if abs(e[m]) <= mpmath.mp.eps * (abs(d[m]) + abs(d[m + 1])):
                    break
                m = m + 1
            if m == l:
                break

            if j >= iterlim:
                raise RuntimeError("tridiag_eigen: no convergence to an eigenvalue after %d iterations" % iterlim)

            j += 1

            # form shift

            p = d[l]
            g = (d[l + 1] - p) / (2 * e[l])
            r = mpmath.mp.hypot(g, 1)

            if g < 0:
                s = g - r
            else:
                s = g + r

            g = d[m] - p + e[l] / s

            s, c, p = 1, 1, 0

            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) > abs(g):             # this here is a slight improvement also used in gaussq.f or acm algorithm 726.
                    c = g / f
                    r = mpmath.mp.hypot(c, 1)
                    e[i + 1] = f * r
                    s = 1 / r
                    c = c * s
                else:
                    s = f / g
                    r = mpmath.mp.hypot(s, 1)
                    e[i + 1] = g * r
                    c = 1 / r
                    s = s * c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

            d[l] = d[l] - p
            e[l] = g
            e[m] = 0

    for ii in range(1, n):
        # sort eigenvalues (bubble-sort)
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] >= p:
                continue
            k = j
            p = d[k]
        if k == i:
            continue
        d[k] = d[i]
        d[i] = p


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
    
    d = a.copy()
    e =  mpmath.matrix(n,1)

    for i in range(n-1):
        e[i] = mpmath.sqrt(b[i+1])

    return d, e

def LobattoMatFrom3Term(a,b,n):
    
    d = a.copy()
    e =  mpmath.matrix(n,1)

    for i in range(n-2):
        e[i] = mpmath.sqrt(b[i+1])

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

    d[n-1] = alpha
    e[n-2] = mpmath.sqrt(beta)

    return d, e

def RadauMatFrom3Term(a,b,n,x):
    
    d = a.copy()
    e =  mpmath.matrix(n,1)
    
    for i in range(n-1):
        e[i] = mpmath.sqrt(b[i+1])

    m = n-1
    phim = EvalAllFrom3Term(a,b,m,x)
    alpha = x - b[m] * phim[m-1] / phim[m]

    d[n-1] = alpha

    return d, e

@functools.cache
def ComputeQuadNodes(n, method = "Gauss"):
    
    if method in all_GL_int:
        
        a, b = ShiftedGaussLegendre3Term(n)

        if method == "Gauss":
            z, e = GaussMatFrom3Term(a,b,n)
        elif method == "Radau_I":
            x = mpmath.mpf(0)
            z, e = RadauMatFrom3Term(a,b,n,x)
        elif method == "Radau_II":
            x = mpmath.mpf(1)
            z, e = RadauMatFrom3Term(a,b,n,x)
        elif method == "Lobatto_III":
            z, e = LobattoMatFrom3Term(a,b,n)
        
        tridiag_eigenvalues(z, e)   
        
    elif method == "Cheb_I":
        
        z = mpmath.matrix(n,1)
        alpha = mpmath.mp.pi / (2*n)
        for i in range(n):
            z[i] = (1 - mpmath.cos(alpha * (2*i+1)))/2
    
    elif method == "Cheb_II":
        
        z = mpmath.matrix(n,1)
        alpha = mpmath.mp.pi / (n+1)
        for i in range(n):
            z[i] = (1 + mpmath.cos(alpha * (n-i)))/2    
            
    elif method == "ClenshawCurtis":
        
        z = mpmath.matrix(n,1)
        alpha = mpmath.mp.pi / (n-1)
        for i in range(n):
            z[i] = (1 + mpmath.cos(alpha * (n-1-i)))/2    
                    
    elif method == "NewtonCotesOpen":
        
        z = mpmath.matrix(n,1)
        alpha = mpmath.fraction(1,n+1)
        for i in range(n):
            z[i] = alpha * (i+1)    
                            
    elif method == "NewtonCotesClosed":
        
        z = mpmath.matrix(n,1)
        alpha = mpmath.fraction(1,n-1)
        for i in range(n):
            z[i] = alpha * i
    
    else:
        raise ValueError(f"Unknown method {method}")

    return z

def ComputeLagrangeWeights(n, xi):
    """
        Computes Lagrange weights for barycentric Lagrange interpolation
    """

    wmat = mpmath.matrix(n)
    
    wmat[0,0] = 1
    
    for i in range(1,n):
        for j in range(i):
            wmat[j,i] = (xi[j] - xi[i]) * wmat[j,i-1]
        
        wmat[i,i] = 1
        for j in range(i):
            wmat[i,i] *= (xi[i] - xi[j])
    
    wi = mpmath.matrix(n,1)
    
    for i in range(n):
        wi[i] = 1 / wmat[i,n-1]

    return wi

def ComputeAvec(n, xi):
    
    amat = mpmath.matrix(n+1)
    avec = mpmath.matrix(n,1)
    
    amat[0,0] = -xi[0]
    amat[1,0] = 1
    
    for j in range(1,n):
        amat[0,j] = -xi[j] * amat[0,j-1]
        
        for i in range(1,j+1):
            amat[i,j] = amat[i-1,j-1]-xi[j]*amat[i,j-1]
        
        amat[j+1,j] = amat[j,j-1]
    
    for j in range(n):
        avec[j] = amat[n-j,n-1] 
        
    return avec
        
def ComputeVandermondeInverseParker(n, xi, wi=None):
    
    if wi is None:
        wi = ComputeLagrangeWeights(n, xi)

    l = n-1
    
    avec = ComputeAvec(n, xi)

    qmat = mpmath.matrix(n)

    for j in range(n):
        qmat[l,j] += 1
        
    for j in range(n):
        for i in range(1,n):
            qmat[l-i,j] += xi[j] * qmat[l-i+1,j] + avec[i]
    
    for i in range(n):
        for j in range(n):
            qmat[l-i,j] *= wi[j]
    
    return qmat

def BuildButcherCMat(z,n):
    
    mat = mpmath.matrix(n)

    for j in range(n):
        for i in range(n):
            mat[j,i] = z[j]**i
            
    return mat

def Build_integration_RHS(z,n):
    
    rhs = mpmath.matrix(1,n)

    for j in range(n):
        rhs[j] = 1
        rhs[j] /= j+1
            
    return rhs

def BuildButcherCRHS(y,z,n,m):
    
    rhs = mpmath.matrix(n,m)

    for j in range(n):
        for i in range(m):
            rhs[j,i] = (z[j]**(i+1) - y[j]**(i+1))/(i+1)
            
    return rhs

def ComputeButcher_collocation(z, vdm_inv, n):
    
    y = mpmath.matrix(n,1)
    for i in range(n):
        y[i] = 0
    
    rhs = BuildButcherCRHS(y,z,n,n)
    Butcher_a = rhs * vdm_inv
    
    zp = mpmath.matrix(n,1)
    for i in range(n):
        y[i]  = 1
        zp[i] = 1+z[i]
        
    rhs = BuildButcherCRHS(y,zp,n,n)
    Butcher_beta = rhs * vdm_inv    
    
    for i in range(n):
        y[i]  = -1 + z[i]
        zp[i] = 0
        
    rhs = BuildButcherCRHS(y,zp,n,n)
    Butcher_gamma = rhs * vdm_inv
    
    return Butcher_a, Butcher_beta, Butcher_gamma

def ComputeButcher_sub_collocation(z, n):

    parker_inv = ComputeVandermondeInverseParker(n-1, z)
    
    y = mpmath.matrix(n,1)
    for i in range(n):
        y[i] = 0
    
    rhs_plus = BuildButcherCRHS(y,z,n,n)
    rhs = rhs_plus[1:n,0:(n-1)]
    Butcher_a_sub = rhs * parker_inv
    
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

def Get_quad_method_from_RK_method(method="Gauss"):
    for quad_method in all_GL_int:
        if quad_method in method:
            return quad_method
    else:
        return method

@functools.cache
def ComputeNamedGaussButcherTables(n, dps=60, method="Gauss"):
    
    mpmath.mp.dps = dps
    quad_method = Get_quad_method_from_RK_method(method)
    w, z, wlag, vdm_inv = ComputeQuadratureTables(n, dps, quad_method)
    
    Butcher_a, Butcher_beta , Butcher_gamma = ComputeButcher_collocation(z, vdm_inv, n)
    
    if method in ["Lobatto_IIIC", "Lobatto_IIIC*", "Lobatto_IIID"]:
        Butcher_a = ComputeButcher_sub_collocation(z,n)
    
    known_method = False
    if method in ["Gauss", "Lobatto_IIIA", "Radau_IIA", "Lobatto_IIIC*", "Cheb_I", "Cheb_II", "ClenshawCurtis", "NewtonCotesOpen", "NewtonCotesClosed"]:
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
        
    if not(known_method):
        raise ValueError(f'Unknown method {method}')

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
    elif "Cheb" in method:
        th_cvg_rate = n + (n % 2)
    elif method == "ClenshawCurtis":
        th_cvg_rate = n + (n % 2)
    elif method == "NewtonCotesOpen":
        th_cvg_rate = n + (n % 2)
    elif method == "NewtonCotesClosed":
        th_cvg_rate = n + (n % 2)
    
    return th_cvg_rate

@functools.cache
def ComputeQuadratureTables(n, dps=30, method="Gauss"):

    mpmath.mp.dps = dps
    z = ComputeQuadNodes(n, method=method)
    return ComputeQuadratureTablesFromNodes(z, dps=30)

def ComputeQuadratureTablesFromNodes(nodes, dps=30):

    mpmath.mp.dps = dps

    n = len(nodes)
    
    if isinstance(nodes, mpmath.matrix):
        z = nodes
    else:
        # Not sure how to cast properly
        z = mpmath.matrix(n,1)
        for i in range(n):
            z[i] = nodes[i]
        
    wlag = ComputeLagrangeWeights(n, z)
    
    vdm_inv = ComputeVandermondeInverseParker(n, z, wlag)
    rhs = Build_integration_RHS(z, n)
    w = rhs * vdm_inv
    
    return w, z, wlag, vdm_inv

def ComputeQuadrature(n=2, dps=30, method="Gauss", nodes=None):
    """Computes a :class:`choreo.segm.quad.QuadTable`

    The computation is performed at a user-defined precision using `mpmath <https://mpmath.org/doc/current>`_ to ensure that the result does not suffer from precision loss, even at relatively high orders.
    
    Available choices for ``method`` are:
    
    * ``"Gauss"``
    * ``"Radau_I"`` and ``"Radau_II"``
    * ``"Lobatto_III"``
    * ``"Cheb_I"`` and ``"Cheb_II"``
    * ``"ClenshawCurtis"``
    * ``"NewtonCotesOpen"`` and ``"NewtonCotesClosed"``
    
    Alternatively, the user can supply an array of node values between ``0.`` and ``1.``. The result is then the associated collocation method.
    
    Example
    -------
    
    >>> import choreo
    >>> choreo.segm.multiprec_tables.ComputeQuadrature(n=2, dps=60, method="Lobatto_III")
    QuadTable object with 2 nodes
    Nodes: [7.7787691e-62 1.0000000e+00]
    Weights: [0.5 0.5]
    Barycentric Lagrange interpolation weights: [-1.  1.]
    >>> 
    >>> choreo.segm.multiprec_tables.ComputeQuadrature(nodes=[0., 0.25, 0.5, 0.75, 1.])
    QuadTable object with 5 nodes
    Nodes: [0.   0.25 0.5  0.75 1.  ]
    Weights: [0.07777778 0.35555556 0.13333333 0.35555556 0.07777778]
    Barycentric Lagrange interpolation weights: [ 10.66666667 -42.66666667  64.         -42.66666667  10.66666667]

    Parameters
    ----------
    n : :class:`python:int`, optional
        Order of the method. By default ``10``.
    dps : :class:`python:int`, optional
        Context precision in `mpmath <https://mpmath.org/doc/current>`_. See :doc:`mpmath:contexts` for more info. By default ``30``.
    method : :class:`python:str`, optional
        Name of the method, by default ``"Gauss"``.
    nodes: :class:`numpy:numpy.ndarray` | :class:`mpmath:mpmath.matrix` | :data:`python:None`, optional
        Array of integration node values. By default, :data:`python:None`.

    Returns
    -------
    :class:`choreo.segm.quad.QuadTable`
        The resulting nodes and weights of the quadrature.
    """    

    if nodes is None:
        th_cvg_rate = GetConvergenceRate(method, n)
        w, z, wlag, vdm_inv = ComputeQuadratureTables(n, dps=dps, method=method)
    else:
        n = len(nodes)
        th_cvg_rate = n
        w, z, wlag, vdm_inv = ComputeQuadratureTablesFromNodes(nodes, dps=dps)

    w_np = np.array(w.tolist(),dtype=np.float64).reshape(n)
    z_np = np.array(z.tolist(),dtype=np.float64).reshape(n)
    w_lag_np = np.array(wlag.tolist(),dtype=np.float64).reshape(n)
    
    return QuadTable(
        w = w_np                    ,
        x = z_np                    ,
        wlag = w_lag_np             ,
        th_cvg_rate = th_cvg_rate   ,
    )

def ComputeImplicitRKTable(n=2, dps=60, method="Gauss", nodes=None):
    """Computes a :class:`choreo.segm.ODE.ImplicitRKTable`

    The computation is performed at a user-defined precision using `mpmath <https://mpmath.org/doc/current>`_ to ensure that the result does not suffer from precision loss, even at relatively high orders.
    
    Available choices for ``method`` are:
    
    * ``"Gauss"``
    * ``"Radau_IA"``, ``"Radau_IIA"``, ``"Radau_IB"`` and ``"Radau_IIB"``
    * ``"Lobatto_IIIA"``, ``"Lobatto_IIIB"``, ``"Lobatto_IIIC"``, ``"Lobatto_IIIC*"``, ``"Lobatto_IIID"`` and ``"Lobatto_IIIS"`` 
    * ``"Cheb_I"`` and ``"Cheb_II"``
    * ``"ClenshawCurtis"``
    * ``"NewtonCotesOpen"`` and ``"NewtonCotesClosed"``
    
    Alternatively, the user can supply an array of node values between ``0.`` and ``1.``. The result is then the associated collocation method. Cf Theorem 7.7 of :footcite:`hairer1987solvingODEI` for more details.
    
    :cited:
    .. footbibliography::
    
    Example
    -------
    
    >>> import choreo
    >>> Gauss = choreo.segm.multiprec_tables.ComputeImplicitRKTable(n=2, dps=60, method="Gauss")
    >>> Gauss
    ImplicitRKTable object of order 2
    >>> Gauss.a_table
    array([[ 0.25      , -0.03867513],
        [ 0.53867513,  0.25      ]])
    >>> Gauss.b_table
    array([0.5, 0.5])
    >>> Gauss.c_table
    array([0.21132487, 0.78867513])
    
    Parameters
    ----------
    n : :class:`python:int`, optional
        Order of the method. By default ``2``.
    dps : :class:`python:int`, optional
        Context precision in `mpmath <https://mpmath.org/doc/current>`_. See :doc:`mpmath:contexts` for more info. By default ``30``.
    method : :class:`python:str`, optional
        Name of the method, by default ``"Gauss"``.
    nodes: :class:`numpy:numpy.ndarray` | :class:`mpmath:mpmath.matrix` | :data:`python:None`, optional
        Array of integration node values. By default, :data:`python:None`.

    Returns
    -------
    :class:`choreo.segm.quad.QuadTable`
        The resulting nodes and weights of the quadrature.
    """    
    

    if nodes is None:

        th_cvg_rate = GetConvergenceRate(method, n)
        
        quad_method = Get_quad_method_from_RK_method(method)
        quad_table = ComputeQuadrature(n, dps=dps, method=quad_method)
        Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = ComputeNamedGaussButcherTables(n, dps=dps, method=method)
        
    else:
        n = len(nodes)
        th_cvg_rate = n
        w, z, wlag, vdm_inv = ComputeQuadratureTablesFromNodes(nodes, dps=dps)
        
        w_np = np.array(w.tolist(),dtype=np.float64).reshape(n)
        z_np = np.array(z.tolist(),dtype=np.float64).reshape(n)
        w_lag_np = np.array(wlag.tolist(),dtype=np.float64).reshape(n)
        
        quad_table = QuadTable(
            w = w_np                    ,
            x = z_np                    ,
            wlag = w_lag_np             ,
            th_cvg_rate = th_cvg_rate   ,
        )
        
        Butcher_a, Butcher_beta , Butcher_gamma = ComputeButcher_collocation(z, vdm_inv, n)
        
    Butcher_a_np = np.array(Butcher_a.tolist(),dtype=np.float64)
    Butcher_beta_np = np.array(Butcher_beta.tolist(),dtype=np.float64)
    Butcher_gamma_np = np.array(Butcher_gamma.tolist(),dtype=np.float64)
    
    return ImplicitRKTable(
        a_table     = Butcher_a_np      ,
        quad_table  = quad_table        ,
        beta_table  = Butcher_beta_np   ,
        gamma_table = Butcher_gamma_np  ,
        th_cvg_rate = th_cvg_rate       ,
    )
        
def ComputeImplicitSymplecticRKTablePair(n=10, dps=60, method="Gauss", nodes=None):

    if nodes is None:

        th_cvg_rate = GetConvergenceRate(method, n)
        
        quad_method = Get_quad_method_from_RK_method(method)
        quad_table = ComputeQuadrature(n, dps=dps, method=quad_method)
        Butcher_a, Butcher_b, Butcher_c, Butcher_beta, Butcher_gamma = ComputeNamedGaussButcherTables(n, dps=dps, method=method)
        
    else:
        n = len(nodes)
        th_cvg_rate = n
        w, z, wlag, vdm_inv = ComputeQuadratureTablesFromNodes(nodes, dps=dps)
        
        w_np = np.array(w.tolist(),dtype=np.float64).reshape(n)
        z_np = np.array(z.tolist(),dtype=np.float64).reshape(n)
        w_lag_np = np.array(wlag.tolist(),dtype=np.float64).reshape(n)
        
        quad_table = QuadTable(
            w = w_np                    ,
            x = z_np                    ,
            wlag = w_lag_np             ,
            th_cvg_rate = th_cvg_rate   ,
        )
        
        Butcher_a, Butcher_beta , Butcher_gamma = ComputeButcher_collocation(z, vdm_inv, n)
        
    Butcher_a_ad = SymplecticAdjointButcher(Butcher_a, Butcher_b, n)  
    
    Butcher_a_np = np.array(Butcher_a.tolist(),dtype=np.float64)
    Butcher_beta_np = np.array(Butcher_beta.tolist(),dtype=np.float64)
    Butcher_gamma_np = np.array(Butcher_gamma.tolist(),dtype=np.float64)
    Butcher_a_ad_np = np.array(Butcher_a_ad.tolist(),dtype=np.float64)
    
    rk = ImplicitRKTable(
        a_table     = Butcher_a_np      ,
        quad_table  = quad_table        ,
        beta_table  = Butcher_beta_np   ,
        gamma_table = Butcher_gamma_np  ,
        th_cvg_rate = th_cvg_rate       ,
    )
    
    rk_ad = ImplicitRKTable(
        a_table     = Butcher_a_ad_np   ,
        quad_table  = quad_table        ,
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
        c_table[i+1]    = val
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
    
    