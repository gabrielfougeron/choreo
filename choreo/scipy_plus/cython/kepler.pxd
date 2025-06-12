cimport numpy as np
cimport cython

cpdef double solve(double M, double ecc) noexcept nogil
cpdef (double, double, double, double, double) kepler(double M, double ecc) noexcept nogil