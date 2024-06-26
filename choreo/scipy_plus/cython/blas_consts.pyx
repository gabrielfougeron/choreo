import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport sqrt as csqrt

cdef char *transn = 'n'
cdef char *transt = 't'

cdef int int_zero = 0
cdef int int_one = 1
cdef int int_two = 2
cdef int int_minusone = -1

cdef double minusone_double = -1
cdef double half_double = 0.5
cdef double two_double = 2.
cdef double one_double = 1.
cdef double zero_double = 0.
cdef double ctwopi = 2*np.pi
cdef double ctwopisqrt2 = ctwopi*csqrt(2.)
cdef double cfourpi = 2*ctwopi
cdef double ctwopisq = 2*np.pi*np.pi
cdef double cfourpisq = ctwopi*ctwopi

cdef double complex zero_double_complex = 0.
cdef double complex one_double_complex = 1.
cdef double complex cminusitwopi = -1j*ctwopi
cdef double complex citwopi = 1j*ctwopi


