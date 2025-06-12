
# Freely adapted / copied from kepler.py by Dan Foreman-Mackey. See LICENSES directory for more info.

import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from choreo.scipy_plus.cython.blas_consts cimport *

from libc.math cimport sqrt as csqrt
from libc.math cimport fmod as cmod
from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport sqrt as csqrt
from libc.math cimport cos as ccos
from libc.math cimport sin as csin

cdef inline double npy_mod(double a, double b) noexcept nogil:

    cdef double mod = cmod(a, b)

    if (b == 0.):
        return mod

    if (mod == 0.):
        return mod
    else:
        if (b < 0) != (mod < 0):
            mod += b
    
    return mod

#  A solver for Kepler's equation based on:
# 
#  Nijenhuis (1991)
#  http://adsabs.harvard.edu/abs/1991CeMDA..51..319N
# 
#  and
# 
#  Markley (1995)
#  http://adsabs.harvard.edu/abs/1995CeMDA..63..101M

cdef double markley_fac1 = 3 * cpi / (cpi - 6 / cpi)
cdef double markley_fac2 = 1.6 / (cpi - 6 / cpi)

@cython.cdivision(True)
cdef inline double markley_starter(double M, double ecc, double ome) noexcept nogil:

    cdef double M2 = M * M
    cdef double alpha = markley_fac1 + markley_fac2 * (cpi - M) / (1. + ecc)
    cdef double d = 3 * ome + alpha * ecc
    cdef double alphad = alpha * d
    cdef double r = (3 * alphad * (d - ome) + M2) * M
    cdef double q = 2 * alphad * ome - M2
    cdef double q2 = q * q
    cdef double w = cpow( cfabs(r) + csqrt(q2 * q + r * r) , 2./3)

    return (2 * r * w / (w * w + w * q + q2) + M) / d

@cython.cdivision(True)
cdef inline double refine_estimate(double M, double ecc, double ome, double E) noexcept nogil:

    cdef double sE = E - csin(E)
    cdef double cE = 1 - ccos(E)

    cdef double f_0 = ecc * sE + E * ome - M
    cdef double f_1 = ecc * cE + ome
    cdef double f_2 = ecc * (E - sE)
    cdef double f_3 = 1 - f_1
    cdef double d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    cdef double d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    cdef double d_42 = d_4 * d_4
    cdef double dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return E + dE

cpdef double solve(double M, double ecc) noexcept nogil:
    """
        Solves Kepler's equation E - ecc * sin(E) = M for E given M and 0. <= ecc < 1.
    """

    cdef double Mp = npy_mod(M, ctwopi)
    cdef double R = M - Mp

    cdef bint high = Mp > cpi
    if high:
        Mp = ctwopi - Mp

    cdef double ome = 1. - ecc
    cdef double E = markley_starter(Mp, ecc, ome)

    E = refine_estimate(Mp, ecc, ome, E)

    if high:
        E = ctwopi - E

    return E + R

@cython.cdivision(True)
cpdef (double, double, double, double, double) kepler(double M, double ecc) noexcept nogil:

    cdef double E = solve(M, ecc)

    cdef double cE = ccos(E)
    cdef double denom = 1. + cE

    cdef double sE, tanf2, tanf2_2
    cdef double cosf, sinf
    cdef double sqrt_alpha, fac, tfp

    sqrt_alpha = csqrt((1+ecc)/(1-ecc))

    if denom > 1e-16:

        sE = csin(E)

        fac = sqrt_alpha  / denom

        tanf2 = fac * sE
        tanf2_2 = tanf2 * tanf2

        tfp = fac  / (1 - ecc * cE)

        denom = 1. / (1 + tanf2_2)
        cosf = (1 - tanf2_2) * denom
        sinf = 2 * tanf2 * denom

        cosfp = (-2) * sinf * denom * tfp
        sinfp = 2 * cosf * denom * tfp

    else:

        cosf = -1.
        sinf = 0.

        cosfp = 0.
        sinfp = -1. / ((1+ecc) * sqrt_alpha)

    return E, cosf, sinf, cosfp, sinfp

