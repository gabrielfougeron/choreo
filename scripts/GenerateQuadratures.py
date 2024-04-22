import mpmath
import numpy as np

import os
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import math as m
import numpy as np
import sys
import time

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 
import pyquickbench

dps = 30
dps_overkill = 1000
mpmath.mp.dps = dps

n_max = 8

def eigenvect(n):

    w, z = choreo.scipy_plus.multiprec_tables.QuadFrom3Term(n)  
    

def vdm(n):
    
    a, b = choreo.scipy_plus.multiprec_tables.ShiftedGaussLegendre3Term(n)
    ww, zz, _ = choreo.scipy_plus.multiprec_tables.QuadFrom3Term_VDM(n)



all_funs = [
    vdm,
    eigenvect   ,
]

all_sizes = [2**n for n in range(n_max)]
 
bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'quadrature_timings.npy'
timings_filename = os.path.join(bench_folder,basename_bench_filename)


all_times = pyquickbench.run_benchmark(
    all_sizes   ,
    all_funs    ,
    filename = timings_filename,
    ForceBenchmark = True,
)

for relative_to in [
    None,
    {pyquickbench.fun_ax_name:"vdm"}
]:
    
    pyquickbench.plot_benchmark(
        all_times   ,
        all_sizes   ,
        all_funs    ,
        relative_to_val = relative_to,
    )

    



# def cp_err(A,B,n):
#     
#     err = 0
#     for i in range(n):
#         err += abs(A[i] - B[i])
#             
#     return float(err)
# 
# for nm1 in range(n_max):
#     n = nm1+1
#     
#     mpmath.mp.dps = dps_overkill
#     a, b = choreo.scipy_plus.multiprec_tables.ShiftedGaussLegendre3Term(n)
#     wo, zo = choreo.scipy_plus.multiprec_tables.QuadFrom3Term(a,b,n)
#     
#     mpmath.mp.dps = dps
#     
#     a, b = choreo.scipy_plus.multiprec_tables.ShiftedGaussLegendre3Term(n)
#     w, z = choreo.scipy_plus.multiprec_tables.QuadFrom3Term(a,b,n)
#     
#     print("Eigenvec")
#     
#     print(cp_err(w,wo,n))
#     print(cp_err(z,zo,n))
#     
#     a, b = choreo.scipy_plus.multiprec_tables.ShiftedGaussLegendre3Term(n)
#     ww, zz, _ = choreo.scipy_plus.multiprec_tables.QuadFrom3Term_VDM(a,b,n)
#     
#     print("VDM solve")
#     
#     print(cp_err(ww,wo,n))
#     print(cp_err(zz,zo,n))
#     print()
#     