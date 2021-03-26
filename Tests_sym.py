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

nbody = 4
ncoeff = 100
mass = np.ones((nbody))

rotangle = 2*np.pi * 1/4

rotmat = np.array([[np.cos(rotangle),-np.sin(rotangle)],[np.sin(rotangle),np.cos(rotangle)]],dtype=np.float64)

Sym_list = []

Sym_list.append(ChoreoSym(
    LoopTarget=0,
    LoopSource=1,
    SpaceRot = rotmat,
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))

Sym_list.append(ChoreoSym(
    LoopTarget=1,
    LoopSource=2,
    SpaceRot = rotmat,
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))


Sym_list.append(ChoreoSym(
    LoopTarget=3,
    LoopSource=2,
    SpaceRot = rotmat,
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))

Sym_list.append(ChoreoSym(
    LoopTarget=1,
    LoopSource=3,
    SpaceRot = rotmat,
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))


TreatSymmetries(nbody,ncoeff,mass,Sym_list=Sym_list)
# ~ TreatSymmetries(nbody,ncoeff,mass,Sym_list=[])
