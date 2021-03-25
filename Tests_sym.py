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

import fractions

from Choreo_funs import *

import networkx as nx

nbody = 3
ncoeff = 100


Sym_list = []

Sym_list.append(ChoreoSym(
    LoopTarget=0,
    LoopSource=1,
    SpaceRot = np.identity(ndim,dtype=np.float64),
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))

Sym_list.append(ChoreoSym(
    LoopTarget=1,
    LoopSource=2,
    SpaceRot = np.identity(ndim,dtype=np.float64),
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))

Sym_list.append(ChoreoSym(
    LoopTarget=2,
    LoopSource=0,
    SpaceRot = np.identity(ndim,dtype=np.float64),
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))

# ~ TreatSymmetries(nbody,ncoeff,Sym_list=Sym_list)
