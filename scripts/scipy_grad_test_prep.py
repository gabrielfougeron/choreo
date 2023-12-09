import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import numpy as np
import math as m
import scipy
import choreo
import time
import matplotlib.pyplot as plt



orderlist = [1,2]
epslist = [10**(-i) for i in range(16)]

def F(x):
    x = np.asarray(x).T
    d = np.diag([3, 2, 1.5, 1, 0.5])
    c = 0.01
    f = -d @ x - c * float(x.T @ x) * x
    return f

F.xin = np.array([1, 1, 1, 1, 1])
F.KNOWN_BAD = {}
F.JAC_KSP_BAD = {}
F.ROOT_JAC_KSP_BAD = {}

def F_GRAD0(x, v):
    x = np.asarray(x).T
    dx = np.asarray(v).T
    d = np.diag([3, 2, 1.5, 1, 0.5])
    c = 0.01

    df = -d @ dx - c * (2*float(x.T @ dx) * x + float(x.T @ x) * dx)
    return df  
    
    


def F3(x):
    A = np.array([[-2, 1, 0.], [1, -2, 1], [0, 1, -2]])
    b = np.array([1, 2, 3.])

    return A @ x - b


def F_GRAD3(x, v):
    A = np.array([[-2, 1, 0.], [1, -2, 1], [0, 1, -2]])
    return A @ v


F3.xin = np.array([1, 2, 3])
F3.KNOWN_BAD = {}
F3.JAC_KSP_BAD = {}
F3.ROOT_JAC_KSP_BAD = {}




def F4_powell(x):
    A = 1e4
    return np.array([A*x[0]*x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - (1 + 1/A)])

def F_GRAD4(x, v):
    A = 1e4
    return np.array([A*(x[0]*v[1] +v[0]*x[1]), -np.exp(-x[0])*v[0] - np.exp(-x[1])*v[1]])


F4_powell.xin = np.array([-1, -2])



def F6(x):
    x1, x2 = x
    J0 = np.array([[-4.256, 14.7],
                   [0.8394989, 0.59964207]])
    v = np.array([(x1 + 3) * (x2**5 - 7) + 3*6,
                  np.sin(x2 * np.exp(x1) - 1)])
    return -np.linalg.solve(J0, v)

def F_GRAD6(x, w):
    x1, x2 = x
    y1, y2 = w
    J0 = np.array([[-4.256, 14.7],
                   [0.8394989, 0.59964207]])
    dv = np.array([
        y1 * (x2**5 - 7) + (x1 + 3) * 5 * x2**4 * y2,
        np.cos(x2 * np.exp(x1) - 1) * (y2 * np.exp(x1) + x2 * np.exp(x1) * y1)
    ])
    return -np.linalg.solve(J0, dv)


F6.xin = np.array([-0.5, 1.4])



all_funs = [F,F3,F4_powell,F6]
all_grad_funs = [F_GRAD0,F_GRAD3,F_GRAD4,F_GRAD6]

dpi = 150
figsize = (1600/dpi, 800 / dpi)

for fun,grad_fun in zip(all_funs, all_grad_funs):
    

    fig, ax = plt.subplots(
        figsize = figsize,
        dpi = dpi   ,
    )


    for i, order in enumerate(orderlist):

        err = choreo.scipy_plus.test.compare_FD_and_exact_grad(fun,grad_fun,fun.xin,epslist=epslist,order=order,vectorize=False)

        plt.plot(epslist,err)
            
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()

    plt.show()