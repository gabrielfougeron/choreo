import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions
import scipy.integrate
import scipy.special

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime



test_names = [
"y'' = -y",
"y'' = - exp(y)",
"y'' = xy",
]

# SymplecticMethod = 'SymplecticEuler'
# SymplecticMethod = 'SymplecticStormerVerlet'

# 
# SymplecticMethod = 'SymplecticEuler_Xfirst'
# SymplecticMethod = 'SymplecticEuler_Vfirst'
# SymplecticMethod = 'SymplecticStormerVerlet_XV'
# SymplecticMethod = 'SymplecticStormerVerlet_VX'

# SymplecticMethod = 'SymplecticEuler_Table_XV'
SymplecticMethod = 'SymplecticEuler_Table_VX'



the_integrators = {SymplecticMethod:choreo.GetSymplecticIntegrator(SymplecticMethod)}

# the_integrators = choreo.all_unique_SymplecticIntegrators

for SymplecticMethod,SymplecticIntegrator in the_integrators.items() :
    print('')
    print('SymplecticMethod : ',SymplecticMethod)


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
            gun = lambda t,x: -x


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


        t_span = (0.,1.)

        ex_init  = ex_sol(t_span[0])
        ex_final = ex_sol(t_span[1])

        x0 = ex_init[0          :  test_ndim]
        v0 = ex_init[test_ndim  :2*test_ndim]



        for nint in [10,100,1000,10000,100000]:

            xf,vf = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint)

            sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))

            print(f'{np.linalg.norm(sol-ex_final):e}')
