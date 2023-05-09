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
    # "exp",
    "cos",
]




methods = ['Gauss']


all_nsteps = range(1,5)


# refinement_lvl = [1,2,4,8,16,32,64,128,256,512,1024,2048]
# refinement_lvl = [1,10,100]
refinement_lvl = list(range(1,11))
# refinement_lvl = list(range(1,101))


for the_test in test_names:

    print('')
    print('test name : ',the_test)

    if the_test == "exp" :
        # WOLFRAM
        # f(x) = exp(x)
        # F(x) = exp(x)

        test_ndim = 1

        fun = lambda x: np.array([m.exp(x)])
        Fun = lambda x: np.array([m.exp(x)])

    if the_test == "cos" :
        # WOLFRAM
        # f(x) = cos(x)
        # F(x) = sin(x)

        test_ndim = 1

        fun = lambda x: np.array([m.cos(x)])
        Fun = lambda x: np.array([m.sin(x)])


    x_span = (0.,10.)
    exact = Fun(x_span[1]) - Fun(x_span[0])

    for method in methods :

        for nsteps in all_nsteps:

            print('')
            print(f'Method : {method}     nsteps : {nsteps}')

            w, x = choreo.ComputeQuadrature_np(method,nsteps)

            for iref in range(len(refinement_lvl)):

                nint = refinement_lvl[iref]

                approx = choreo.IntegrateOnSegment(
                    fun = fun,
                    ndim = test_ndim,
                    x_span = x_span,
                    nint = nint,
                    w = w,
                    x = x,
                    nsteps = nsteps,
                )

                error = np.linalg.norm(approx-exact)/np.linalg.norm(exact)

                if (iref > 0):
                    error_mul = max(error/error_prev,1e-16)
                    est_order = -m.log(error_mul)/m.log(refinement_lvl[iref]/refinement_lvl[iref-1])

                    print(f'error : {error:e}     error mul : {error_mul:e}     estimated order : {est_order:.2f}')

                error_prev = error
