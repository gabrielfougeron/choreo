cimport numpy as np
cimport cython
from choreo.segm.cython.quad cimport QuadTable

@cython.final
cdef class ExplicitSymplecticRKTable():

    cdef double[::1] _c_table
    cdef double[::1] _d_table
    cdef Py_ssize_t _th_cvg_rate    # Self-reported convergence rate on smooth functions

    cpdef ExplicitSymplecticRKTable symmetric_adjoint(self)
    cdef double _symmetry_default(self, ExplicitSymplecticRKTable other) noexcept nogil
    cdef bint _is_symmetric_pair(self, ExplicitSymplecticRKTable other, double tol) noexcept nogil

@cython.final
cdef class ImplicitRKTable:

    cdef double[:,::1] _a_table             # A Butcher table.
    cdef QuadTable _quad_table              # Underlying integration method. Contains B Butcher (= _quad_table._w) table and C Butcher tables (= _quad_table._x)
    cdef double[:,::1] _beta_table          # Beta Butcher table for initial guess in convergence loop. 
    cdef double[:,::1] _gamma_table         # Beta Butcher table of the symmetric adjoint.
    cdef Py_ssize_t _th_cvg_rate            # Theoretical convergence rate of the method.


    cpdef ImplicitRKTable symmetric_adjoint(self)
    cdef double _symmetry_default(self, ImplicitRKTable other) noexcept nogil
    cdef bint _is_symmetric_pair(self, ImplicitRKTable other, double tol) noexcept nogil

    cpdef ImplicitRKTable symplectic_adjoint(self)
    cdef double _symplectic_default(self, ImplicitRKTable other) noexcept nogil
    cpdef bint _is_symplectic_pair(self, ImplicitRKTable other, double tol) noexcept nogil