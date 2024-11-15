import os
import sys

import numpy as np
import math as m

import scipy
import choreo


geodim = 2
nbody = 3
bodymass = np.array([1., 2., 3.])
bodycharge = np.array([4., 5., 6.])
Sym_list = []

inter_pow = all_kwargs["inter_pow"]
inter_pm = all_kwargs["inter_pm"]

inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
NBS = choreo.cython._NBodySyst.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list, inter_law)

inter_law = choreo.numba_funs.pow_inter_law(inter_pow/2, inter_pm)