import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import concurrent.futures
# import multiprocessing
import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions
import functools

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime

One_sec = 1e9

test_names = [
# "y'' = -y",
# "y'' = - exp(y)",
# "y'' = xy",
"y' = Az; z' = By",
]




methods = ['SymplecticGauss'+str(i) for i in range(1,10)]

# methods.append('SymplecticEuler')
# methods.append('SymplecticStormerVerlet')

# 
# methods.append('SymplecticEuler_XV')
# methods.append('SymplecticEuler_VX')
# methods.append('SymplecticStormerVerlet_XV')
# methods.append('SymplecticStormerVerlet_VX')

# methods.append('SymplecticEuler_Table_XV')
# methods.append(SymplecticEuler_Table_VX')



the_integrators = {method:choreo.GetSymplecticIntegrator(method) for method in methods}


# the_integrators = choreo.all_unique_SymplecticIntegrators


for the_test in test_names:

    print('')
    print('test name : ',the_test)

    if the_test == "y'' = -y" :
            # WOLFRAM
            # y'' = - y
            # y(x) = A cos(x) + B sin(x)

        test_ndim = 2

        ex_sol = lambda t : np.array( [ np.cos(t) , np.sin(t),-np.sin(t),np.cos(t) ]  )

        fun = lambda t,y:  y
        gun = lambda t,x:  -x

    if the_test == "y'' = - exp(y)" :
            # WOLFRAM
            # y'' = - exp(y)
            # y(x) = - 2 * ln( cosh(t / sqrt(2) ))

        test_ndim = 1

        invsqrt2 = 1./np.sqrt(2.)
        sqrt2 = np.sqrt(2.)
        ex_sol = lambda t : np.array( [ -2*np.log(np.cosh(invsqrt2*t)) , -sqrt2*np.tanh(invsqrt2*t) ]  )

        fun = lambda t,y:  y
        gun = lambda t,x: -np.exp(x)

    if the_test == "y'' = xy" :

        # Solutions: Airy functions
        # Nonautonomous linear test case

        test_ndim = 2

        def ex_sol(t):

            ai, aip, bi, bip = scipy.special.airy(t)

            return np.array([ai,bi,aip,bip])


        fun = lambda t,y:  y
        gun = lambda t,x: np.array([t*x[0],t*x[1]],dtype=np.float64)
        
    if the_test == "y' = Az; z' = By" :

        test_ndim = 2

        A = np.random.rand(test_ndim,test_ndim)
        A = A + A.T
        B = np.random.rand(test_ndim,test_ndim)
        B = B + B.T

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


    for SymplecticMethod,SymplecticIntegrator in the_integrators.items() :
        print('')
        print('SymplecticMethod : ',SymplecticMethod)

        SymplecticIntegrator = functools.partial(SymplecticIntegrator, maxiter = 1000)

        t_span = (0.,np.pi)

        ex_init  = ex_sol(t_span[0])
        ex_final = ex_sol(t_span[1])

        x0 = ex_init[0          :  test_ndim]
        v0 = ex_init[test_ndim  :2*test_ndim]

        # print(x0)
        # print(v0)

        # refinement_lvl = [1,2,4,8,16,32,64,128,256,512,1024,2048]
        # refinement_lvl = [1,10,100]
        refinement_lvl = [1,2,3,4,5,6,7,8,9,10]

        for iref in range(len(refinement_lvl)):

            nint = refinement_lvl[iref]

            x0 = np.copy(ex_init[0          :  test_ndim])
            v0 = np.copy(ex_init[test_ndim  :2*test_ndim])

            t_beg= time.perf_counter_ns()
            xf,vf = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint,nint)
            t_end = time.perf_counter_ns()

            sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))
            error = np.linalg.norm(sol-ex_final)

            # print(f'error : {error:e} time : {(t_end-t_beg)/One_sec:f}')

            if (iref > 0):
                error_mul = max(error/error_prev,1e-16)
                est_order = -m.log(error_mul)/m.log(refinement_lvl[iref]/refinement_lvl[iref-1])

                print(f'error : {error:e}     error mul : {error_mul:e}     estimated order : {est_order:.2f}     time : {(t_end-t_beg)/One_sec:f}')
                # print(f'error : {error:e}     estimated order : {est_order:.2f}')

            error_prev = error
