import math as m
import numpy as np
import scipy
import ctypes

import choreo.segm.cython.test

from choreo.segm.ODE import SymplecticIVP, ImplicitSymplecticIVP
from choreo.segm.multiprec_tables import ComputeQuadrature
from choreo.segm.quad import IntegrateOnSegment

def Quad_cpte_error_on_test(
    fun_name    = "exp"     ,
    quad_method = "Gauss"   ,
    quad_nsteps = 1         ,
    nint        = 1         ,
    DoEFT       = False     ,
):

    if fun_name == "exp" :
        # WOLFRAM
        # f(x) = y*exp(y*x)
        # F(x) = exp(y*x)

        test_ndim = 20
        
        test_ndim_array = np.array([test_ndim], dtype=np.intp)
        user_data = test_ndim_array.ctypes.data_as(ctypes.c_void_p)

        fun = scipy.LowLevelCallable.from_cython(
                    choreo.segm.cython.test ,
                    "exp_fun"               ,
                    user_data               ,
                )

        # fun = lambda x: np.array([y*m.exp(y*x) for y in range(test_ndim)])
        Fun = lambda x: np.array([m.exp(y*x) for y in range(test_ndim)])
        
        x_span = (0.,1.)
        exact = Fun(x_span[1]) - Fun(x_span[0])

    quad = ComputeQuadrature(quad_nsteps, dps = 60, method = quad_method)

    approx = IntegrateOnSegment(
        fun = fun       ,
        ndim = test_ndim,
        x_span = x_span ,
        quad = quad     ,
        nint = nint     ,
        DoEFT = DoEFT   ,
    )

    error = np.linalg.norm(approx-exact)/np.linalg.norm(exact)

    return error
 
def ODE_define_test(eq_name):
     
    if eq_name == "y'' = -y" :
        # WOLFRAM
        # y'' = - y
        # y(x) = A cos(x) + B sin(x)

        test_ndim = 2

        ex_sol = lambda t : np.array( [ np.cos(t) , np.sin(t), -np.sin(t), np.cos(t) ]  )

        fun = lambda t,y:   np.asarray(y)
        gun = lambda t,x:  -np.asarray(x)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] =  xy[2]
            fxy[1] =  xy[3]
            fxy[2] = -xy[0]
            fxy[3] = -xy[1]
            
            return fxy

    if eq_name == "y'' = - exp(y)" :
        # WOLFRAM
        # y'' = - exp(y)
        # y(x) = - 2 * ln( cosh(t / sqrt(2) ))

        test_ndim = 1

        invsqrt2 = 1./np.sqrt(2.)
        sqrt2 = np.sqrt(2.)
        ex_sol = lambda t : np.array( [ -2*np.log(np.cosh(invsqrt2*t)) , -sqrt2*np.tanh(invsqrt2*t) ]  )

        fun = lambda t,y:  np.array(y)
        gun = lambda t,x: -np.exp(x)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] = xy[1]
            fxy[1] = -np.exp(xy[0])

            return fxy

    if eq_name == "y'' = xy" :

        # Solutions: Airy functions
        # Nonautonomous linear test case

        test_ndim = 2

        def ex_sol(t):

            ai, aip, bi, bip = scipy.special.airy(t)

            return np.array([ai,bi,aip,bip])

        fun = lambda t,y: np.array(y)
        gun = lambda t,x: np.array([t*x[0],t*x[1]],dtype=np.float64)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] =  xy[2]
            fxy[1] =  xy[3]
            fxy[2] = t*xy[0]
            fxy[3] = t*xy[1]
            
            return fxy
        
    if eq_name == "y' = Az; z' = By" :

        test_ndim = 10

        A = np.diag(np.array(range(test_ndim)))
        B = np.identity(test_ndim)

        AB = np.zeros((2*test_ndim,2*test_ndim))
        AB[0:test_ndim,test_ndim:2*test_ndim] = A
        AB[test_ndim:2*test_ndim,0:test_ndim] = B

        yo = np.array(range(test_ndim))
        zo = np.array(range(test_ndim))

        yzo = np.zeros(2*test_ndim)
        yzo[0:test_ndim] = yo
        yzo[test_ndim:2*test_ndim] = zo

        def ex_sol(t):
            return scipy.linalg.expm(t*AB).dot(yzo)

        fun = lambda t,z: A.dot(z)
        gun = lambda t,y: B.dot(y)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[:test_ndim] =  A.dot(xy[test_ndim:])
            fxy[test_ndim:] =  B.dot(xy[:test_ndim])

            return fxy

    return fun, gun, fgun, ex_sol, test_ndim
 
def ODE_cpte_error_on_test(
    eq_name     ,
    rk_method   ,
    nint        ,
    **kwargs    ,
):

    fun, gun, fgun, ex_sol, test_ndim = ODE_define_test(eq_name)

    t_span = (0.,np.pi)

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    x0 = ex_init[0          :  test_ndim].copy()
    v0 = ex_init[test_ndim  :2*test_ndim].copy()

    xf,vf = SymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk = rk_method  ,
        nint = nint     ,
        **kwargs        ,
    )
        
    sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))
    error = np.linalg.norm(sol-ex_final)/np.linalg.norm(ex_final)

    return error

def ISPRK_ODE_cpte_error_on_test(
    eq_name     ,
    rk_x        ,
    rk_v        ,
    nint        ,
    **kwargs    ,
):

    fun, gun, fgun, ex_sol, test_ndim = ODE_define_test(eq_name)

    t_span = (0.,np.pi)

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    x0 = ex_init[0          :  test_ndim].copy()
    v0 = ex_init[test_ndim  :2*test_ndim].copy()

    xf,vf = ImplicitSymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk_x = rk_x     ,
        rk_v = rk_v     ,
        nint = nint     ,
        **kwargs        ,
    )
                
    sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))
    error = np.linalg.norm(sol-ex_final)/np.linalg.norm(ex_final)

    return error

def scipy_ODE_cpte_error_on_test(
    eq_name     ,
    method      ,
    nint        ,
    **kwargs    ,
):

    fun, gun, fgun, ex_sol, test_ndim = ODE_define_test(eq_name)

    t_span = (0.,np.pi)
    
    max_step = (t_span[1] - t_span[0]) / nint

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    bunch = scipy.integrate.solve_ivp(
        fun = fgun                      ,
        t_span = t_span                 ,
        y0 = ex_init                    ,
        method = method                 ,
        t_eval = np.array([t_span[1]])  ,
        first_step = max_step           ,
        max_step = max_step             ,
        atol = 1.             ,
        rtol = 1.             ,
    )

    error = np.linalg.norm(bunch.y[:,0]-ex_final)/np.linalg.norm(ex_final)

    return error

def compute_FD(fun,xo,dx,eps,fo=None,order=1):
    
    if fo is None:
        fo = fun(xo)
        
    if order == 1:
        
        xp = xo + eps*dx
        fp = fun(xp)
        dfdx = (fp-fo)/eps
        
    elif (order == 2):
        
        xp = xo + eps*dx
        fp = fun(xp)        
        xm = xo - eps*dx
        fm = fun(xm)
        dfdx = (fp-fm)/(2*eps)
        
    else:
        
        raise ValueError(f"Invalid order {order}")

    return dfdx

def compare_FD_and_exact_grad(fun, gradfun, xo, dx=None, epslist=None, order=1, vectorize=True, relative=True):
    
    if epslist is None:
        epslist = [10**(-i) for i in range(16)]
        
    if dx is None:
        dx = np.array(np.random.rand(*xo.shape), dtype= xo.dtype)
    
    fo = fun(xo)
    if vectorize:
        dfdx_exact = gradfun(xo,dx.reshape(-1,1)).reshape(-1)
    else:
        dfdx_exact = gradfun(xo,dx)
    dfdx_exact_magn = np.linalg.norm(dfdx_exact)
    
    error_list = []
    for eps in epslist:
        dfdx_FD = compute_FD(fun,xo,dx,eps,fo=fo,order=order)
        
        if relative:
            error = np.linalg.norm(dfdx_FD - dfdx_exact) / dfdx_exact_magn 
        else:
            error = np.linalg.norm(dfdx_FD - dfdx_exact)
            
        error_list.append(error)
    
    return np.array(error_list)
        
# Adapted from a base implementation of Algorithm 6.1 of [1] available at https://github.com/python/cpython/blob/main/Lib/test/test_math.py
# [1] Ogita, T., Rump, S. M., & Oishi, S. I. (2005). Accurate sum and dot product. SIAM Journal on Scientific Computing, 26(6), 1955-1988.

import operator
from fractions import Fraction
from itertools import starmap
from collections import namedtuple
from math import log2, exp2, fabs
from random import choices, uniform, shuffle
from statistics import median

def SumExact(x, n=None):
    if n is None:
        return sum(map(Fraction, x))
    else:
        return sum(map(Fraction, x[:n]))

def SumCondition(x):
    return 2.0 * SumExact(map(abs, x)) / abs(SumExact(x))

def SumCondition_given_ex(x, ex):
    return 2.0 * SumExact(map(abs, x)) / abs(ex)

def DotExact(x, y, n=None):
    
    if n is None:
        vec1 = map(Fraction, x)
        vec2 = map(Fraction, y)
    else:
        vec1 = map(Fraction, x[:n])
        vec2 = map(Fraction, y[:n])
        
    return sum(starmap(operator.mul, zip(vec1, vec2, strict=True)))

def DotCondition(x, y):
    return 2.0 * DotExact(map(abs, x), map(abs, y)) / abs(DotExact(x, y))

def linspace(lo, hi, n):
    width = (hi - lo) / (n - 1)
    return [lo + width * i for i in range(n)]

def GenDot(n, c):
    """ Algorithm 6.1 (GenDot) works as follows. The condition number (5.7) of
    the dot product xT y is proportional to the degree of cancellation. In
    order to achieve a prescribed cancellation, we generate the first half of
    the vectors x and y randomly within a large exponent range. This range is
    chosen according to the anticipated condition number. The second half of x
    and y is then constructed choosing xi randomly with decreasing exponent,
    and calculating yi such that some cancellation occurs. Finally, we permute
    the vectors x, y randomly and calculate the achieved condition number.
    """

    assert n >= 6
    assert c >= 2.
    
    n2 = n // 2
    x = [0.0] * n
    y = [0.0] * n
    b = log2(c)

    # First half with exponents from 0 to |_b/2_| and random ints in between
    
    e = choices(range(int(b/2)), k=n2)
    e[0] = int(b / 2) + 1
    e[-1] = 0.0

    x[:n2] = [uniform(-1.0, 1.0) * exp2(p) for p in e]
    y[:n2] = [uniform(-1.0, 1.0) * exp2(p) for p in e]

    dot_exact = DotExact(x, y, n2)
    # Second half
    e = list(map(round, linspace(b/2, 0.0 , n-n2)))
    for i in range(n2, n):
        x[i] = uniform(-1.0, 1.0) * exp2(e[i - n2])
        y[i] = (uniform(-1.0, 1.0) * exp2(e[i - n2]) - dot_exact) / x[i]
        
        dot_exact += Fraction(x[i]) * Fraction(y[i])

    return np.array(x), np.array(y), dot_exact

def GenSum(n,c):

    assert n >= 6
    assert c >= 2.

    n2 = n // 2
    x = [0.0] * n
    b = log2(c)

    e = choices(range(int(b)), k=n2)
    e[0] = int(b) + 1
    e[-1] = 0.0

    x[:n2] = [uniform(-1.0, 1.0) * exp2(p) for p in e]
    
    sum_exact = SumExact(x, n2)
    # Second half
    e = list(map(round, linspace(b/2, 0.0 , n-n2)))
    for i in range(n2, n):
        x[i] = uniform(-1.0, 1.0) * exp2(e[i - n2]) - sum_exact
        sum_exact += Fraction(x[i])
        
    return np.array(x), sum_exact
