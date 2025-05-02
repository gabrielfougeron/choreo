"""
Convergence analysis of implicit Runge-Kutta methods for ODE IVP
================================================================
"""

# %%
# TODO: Redo this example using more modern features of pyquickbench (like plot_intent) to avoid loops 
# Evaluation of relative quadrature error with the following parameters:

# sphinx_gallery_start_ignore

import os
import sys
import itertools
import functools

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy

import choreo
import choreo.segm.precomputed_tables as precomputed_tables

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'ImplicitRK_ivp_cvg_bench_'

# ForceBenchmark = True
ForceBenchmark = False

# sphinx_gallery_end_ignore

method_names = [
    "Gauss"             ,
    "Radau_IA"          ,
    "Radau_IIA"         ,
    "Radau_IB"          ,
    "Radau_IIB"         ,
    "Lobatto_IIIA"      ,
    "Lobatto_IIIB"      ,
    "Lobatto_IIIC"      ,
    "Lobatto_IIIC*"     ,
    "Lobatto_IIID"      ,            
    "Lobatto_IIIS"      ,     
    "Cheb_I"            ,
    "Cheb_II"           ,
    "ClenshawCurtis"    ,
    "NewtonCotesOpen"   ,
    "NewtonCotesClosed" ,      
]

for method in method_names:
    
    print(f'{method = }')
    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method=method, dps=1000)
    
    print(f'rk.a_table = ')
    print(rk.a_table)
    print(f'{rk.c_table = }')
    print(f'{rk.quad_table.w = }')
    # print(rk.nsteps == rk.n_eff_steps)
    print(f'{rk.nsteps = }')
    print(f'{rk.n_eff_steps_updt = }')
    print(f'{rk.n_eff_steps_eval = }')
    
    print()
    