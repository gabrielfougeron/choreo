import numpy as np
import math as m
import scipy.optimize as opt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy
import os, shutil
import time

from Choreo_funs import *


def f(x):
    return (x*x).sum()

d = 1
nvals_list = [10000,20000]
ex_int = 1/3



sampler = UniformRandom(d=d)
# ~ sampler = Halton(d=d)


approx_int = 0

rel_err_list = []

for nvals in nvals_list:
    for i in range(nvals):
        approx_int += f(sampler.random())
        
    approx_int /= nvals
    
    rel_err = abs(approx_int-ex_int)/(ex_int)
    rel_err_list.append(rel_err)
    print('Uniform Random rel error : ',rel_err)




for i in range(len(rel_err_list)-1):
    print(rel_err_list[i]/rel_err_list[i+1])

