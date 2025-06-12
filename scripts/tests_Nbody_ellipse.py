import numpy as np
import cmath
import math

import choreo

from choreo.scipy_plus.cython.kepler import kepler

import pyquickbench


ecc = 0.5
M = np.pi

E, cosf, sinf, dcosf, dsinf = kepler(M, ecc)

print(E)
print(cosf)
print(sinf)
print(dcosf)
print(dsinf)
print()

dM = 1e-5

Ep, cosfp, sinfp, dcosfp, dsinfp = kepler(M+dM, ecc)
Em, cosfm, sinfm, dcosfm, dsinfm = kepler(M-dM, ecc)

dcosf_fd = (cosfp - cosfm)/(2*dM)
dsinf_fd = (sinfp - sinfm)/(2*dM)

print()
print(dcosf)
print(dcosf_fd)

print()
print(dsinf)
print(dsinf_fd)

print()
print(abs(dcosf_fd - dcosf))
print(abs(dsinf_fd - dsinf))

