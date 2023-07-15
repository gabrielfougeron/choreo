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

    for geodim in Physical_dims.all_geodims:
        for nbody in Few_bodies.all_nbodies:

            Id = choreo.ActionSym.Identity(nbody, geodim)

            assert Id.IsIdentity(atol = float64_tols.atol)

            InvId = Id.Inverse()

            assert Id.IsSame(InvId, atol = float64_tols.atol)

def test_Random(float64_tols, Physical_dims, Few_bodies):

    for geodim in Physical_dims.all_geodims:
        for nbody in Few_bodies.all_nbodies:

            Id = choreo.ActionSym.Identity(nbody, geodim)

            perm = np.random.permutation(nbody)

            mat = np.random.random_sample((geodim,geodim))
            sksymmat = mat - mat.T
            rotmat = scipy.linalg.expm(sksymmat)

            timerev = 1 if np.random.random_sample() < 0.5 else -1

            maxden = 10*nbody
            den = np.random.randint(low = 1, high = maxden)
            num = np.random.randint(low = 0, high =    den)

            timeshift = fractions.Fraction(numerator = num, denominator = den)

            A = choreo.ActionSym(
                BodyPerm = perm,
                SpaceRot = rotmat,
                TimeRev = timerev,
                TimeShift = timeshift,
            )

            AInv = A.Inverse()

            assert Id.IsSame(A.Compose(AInv), atol = float64_tols.atol)
            assert Id.IsSame(AInv.Compose(A), atol = float64_tols.atol)