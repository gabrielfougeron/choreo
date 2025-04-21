cimport numpy as np
cimport cython

@cython.final
cdef class QuadTable():
   
    cdef double[::1] _w             # Integration weights on [0,1]
    cdef double[::1] _x             # Integration nodes on [0,1]
    cdef double[::1] _wlag          # Barycentric Lagrange interpolation weights
    cdef Py_ssize_t _th_cvg_rate    # Self-reported convergence rate on smooth functions

    cpdef QuadTable symmetric_adjoint(self)
    cdef double _symmetry_default(self, QuadTable other) noexcept nogil
    cdef bint _is_symmetric_pair(self, QuadTable other, double tol) noexcept nogil