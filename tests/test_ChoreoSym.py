import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import numpy as np
import scipy
import fractions
import choreo

# 
# @ProbabilisticTest()
# def test_Random(float64_tols, Physical_dims, Few_bodies):
# 
#     print("Testing group properties on random transformations.")
# 
#     for geodim in Physical_dims.all_geodims:
#         for nbody in Few_bodies.all_nbodies:
# 
#             print(f"geodim = {geodim}, nbody = {nbody}")
# 
#             A = choreo.ChoreoSym.Random(nbody, geodim)
#             AInv = A.Inverse()
# 
#             assert (A.Compose(AInv)).IsIdentity(atol = float64_tols.atol)
#             assert (AInv.Compose(A)).IsIdentity(atol = float64_tols.atol)
# 
#             B = choreo.ChoreoSym.Random(nbody, geodim)
# 
#             assert not(A.IsSame(B, atol = float64_tols.atol))
# 
# @RepeatTest()
# def test_Modulo_Numerator(float64_tols, Physical_dims, Few_bodies):
# 
#     print("Testing acceptable values of TimeShift numerator.")
# 
#     for geodim in Physical_dims.all_geodims:
#         for nbody in Few_bodies.all_nbodies:
# 
#             print(f"geodim = {geodim}, nbody = {nbody}")
# 
#             A = choreo.ChoreoSym.Random(nbody, geodim)
#             B = choreo.ChoreoSym.Random(nbody, geodim)
# 
#             Ainv = A.Inverse()
#             Binv = B.Inverse()
# 
#             AB = A.ComposeLight(B)
#             BA = B.ComposeLight(A)
# 
#             for Sym in [A,B,AB,BA,Ainv,Binv]:
# 
#                 assert Sym.TimeShift.numerator >= 0
#                 assert Sym.TimeShift.numerator < Sym.TimeShift.denominator