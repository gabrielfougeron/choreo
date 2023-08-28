import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import functools
import itertools
import math as m
import numpy as np
import scipy.linalg
import sys
import matplotlib.pyplot as plt

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 



def cpte_error(
    eq_name,
    rk_method,
    mode,
    nint,
):


    if eq_name == "y'' = -y" :
        # WOLFRAM
        # y'' = - y
        # y(x) = A cos(x) + B sin(x)

        test_ndim = 2

        ex_sol = lambda t : np.array( [ np.cos(t) , np.sin(t),-np.sin(t), np.cos(t) ]  )

        fun = lambda t,y:   y.copy()
        gun = lambda t,x:  -np.array(x.copy())

    if eq_name == "y'' = - exp(y)" :
        # WOLFRAM
        # y'' = - exp(y)
        # y(x) = - 2 * ln( cosh(t / sqrt(2) ))

        test_ndim = 1

        invsqrt2 = 1./np.sqrt(2.)
        sqrt2 = np.sqrt(2.)
        ex_sol = lambda t : np.array( [ -2*np.log(np.cosh(invsqrt2*t)) , -sqrt2*np.tanh(invsqrt2*t) ]  )

        fun = lambda t,y:  y.copy()
        gun = lambda t,x: -np.exp(x)

    if eq_name == "y'' = xy" :

        # Solutions: Airy functions
        # Nonautonomous linear test case

        test_ndim = 2

        def ex_sol(t):

            ai, aip, bi, bip = scipy.special.airy(t)

            return np.array([ai,bi,aip,bip])

        fun = lambda t,y: y.copy()
        gun = lambda t,x: np.array([t*x[0],t*x[1]],dtype=np.float64)
        
    if eq_name == "y' = Az; z' = By" :

        test_ndim = 100

        # A = np.random.rand(test_ndim,test_ndim)
        A = np.identity(test_ndim)
        # A = A + A.T
        # B = np.random.rand(test_ndim,test_ndim)
        B = np.identity(test_ndim)
        # B = B + B.T

        AB = np.zeros((2*test_ndim,2*test_ndim))
        AB[0:test_ndim,test_ndim:2*test_ndim] = A
        AB[test_ndim:2*test_ndim,0:test_ndim] = B

        yo = np.random.rand(test_ndim)
        zo = np.random.rand(test_ndim)

        yzo = np.zeros(2*test_ndim)
        yzo[0:test_ndim] = yo
        yzo[test_ndim:2*test_ndim] = zo

        def ex_sol(t):

            return scipy.linalg.expm(t*AB).dot(yzo)

        fun = lambda t,z: A.dot(z)
        gun = lambda t,y: B.dot(y)



    t_span = (0.,np.pi)

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    x0 = ex_init[0          :  test_ndim].copy()
    v0 = ex_init[test_ndim  :2*test_ndim].copy()
    
#     print(x0.shape)
#     print(v0.shape)
#     print(fun(0.,x0).shape)
#     print(gun(0.,v0).shape)
# 
#     exit()

    xf,vf = choreo.scipy_plus.ODE.ExplicitSymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        nint = nint     ,
        rk = rk_method  ,
        mode = mode     ,
    )

    sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))
    error = np.linalg.norm(sol-ex_final)/np.linalg.norm(ex_final)

    return error


eq_names = [
    "y'' = -y",
    "y'' = - exp(y)",
    "y'' = xy",
    "y' = Az; z' = By",
]

rk_tables = {
    "SymplecticEuler": choreo.scipy_plus.precomputed_tables.SymplecticEuler ,
    "StormerVerlet": choreo.scipy_plus.precomputed_tables.StormerVerlet     ,
    "Ruth3": choreo.scipy_plus.precomputed_tables.Ruth3                     ,
    "Ruth4": choreo.scipy_plus.precomputed_tables.Ruth4                     ,
    "Ruth4Rat": choreo.scipy_plus.precomputed_tables.Ruth4Rat               ,
}

all_nint = [1,2,4,8,16,32,64,128,256,512,1024,2048]

    
all_benchs = {
    eq_name : {
        f'{rk_name} {mode}' : functools.partial(
            cpte_error ,
            eq_name    ,
            rk_table   ,
            mode       ,
        ) for (rk_name, rk_table), mode in itertools.product(rk_tables.items(), ['XV','VX'])
    } for eq_name in eq_names
}






for bench_name, all_funs in all_benchs.items():

    print()
    print(bench_name)

    for rk_name, fun in all_funs.items():

        print()
        print(f'SymplecticMethod :  {rk_name}')

        for iref in range(len(all_nint)):

            nint = all_nint[iref]

            error = fun(nint)
 
            if (iref > 0):
                error_mul = max(error/error_prev,1e-16)
                est_order = -m.log(error_mul)/m.log(all_nint[iref]/all_nint[iref-1])

                print(f'{nint:4d}  error : {error:e}     error mul : {error_mul:e}     estimated order : {est_order:.2f}')
                # print(f'error : {error:e}     estimated order : {est_order:.2f}')

            error_prev = error



# 