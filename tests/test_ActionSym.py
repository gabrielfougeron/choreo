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

def test_Identity(float64_tols, Physical_dims, Few_bodies):

    print("Testing group properties on identity.")

    for geodim in Physical_dims.all_geodims:
        for nbody in Few_bodies.all_nbodies:

            print(f"geodim = {geodim}, nbody = {nbody}")

            Id = choreo.ActionSym.Identity(nbody, geodim)

            assert Id.IsIdentity(atol = float64_tols.atol)

            Id2 = Id.Compose(Id)

            assert Id2.IsIdentity(atol = float64_tols.atol)

            InvId = Id.Inverse()

            assert Id.IsSame(InvId, atol = float64_tols.atol)

@ProbabilisticTest()
def test_Random(float64_tols, Physical_dims, Few_bodies):

    print("Testing group properties on random transformations.")

    for geodim in Physical_dims.all_geodims:
        for nbody in Few_bodies.all_nbodies:

            print(f"geodim = {geodim}, nbody = {nbody}")

            Id = choreo.ActionSym.Identity(nbody, geodim)

            A = choreo.ActionSym.Random(nbody, geodim)
            AInv = A.Inverse()

            assert Id.IsSame(A.Compose(AInv), atol = float64_tols.atol)
            assert Id.IsSame(AInv.Compose(A), atol = float64_tols.atol)

            B = choreo.ActionSym.Random(nbody, geodim)

            assert not(A.IsSame(B, atol = float64_tols.atol))

            assert False