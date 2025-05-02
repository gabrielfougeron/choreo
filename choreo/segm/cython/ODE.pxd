cimport numpy as np
cimport cython
from choreo.segm.cython.quad cimport QuadTable

@cython.final
cdef class ExplicitSymplecticRKTable():

    cdef double[::1] _c_table               # Component of A butcher table for explicit symplectic RK
    cdef double[::1] _d_table               # Component of A butcher table for explicit symplectic RK
    cdef Py_ssize_t _th_cvg_rate            # Self-reported convergence rate on smooth functions

    cdef bint[::1] _cant_skip_f_eval        # Whether calls to fun can be skipped or not
    cdef bint[::1] _cant_skip_x_updt        # Whether update to x can be skipped or not
    cdef bint[::1] _cant_skip_g_eval        # Whether calls to gun can be skipped or not
    cdef bint[::1] _cant_skip_v_updt        # Whether update to x can be skipped or not

    cdef public Py_ssize_t n_eff_steps
    """ Number of effective steps of the method """

    cdef bint _separate_res_buf             # Whether to separate res_x and _res_v

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

#     # TODO: Maybe finish this someday
#     cdef bint[::1] _cant_skip_updt          # Whether the dXV update can be skipped in convergence loop
#     cdef bint[::1] _cant_skip_eval          # Whether calls to fun/gun can be skipped or not
# 
#     cdef public Py_ssize_t n_eff_steps_updt
#     """ Number of effective steps of the method """
#     cdef public Py_ssize_t n_eff_steps_eval
#     """ Number of effective steps of the method """
# 
#     cdef int vec_eval_beg
#     cdef int vec_eval_end

    cpdef ImplicitRKTable symmetric_adjoint(self)
    cdef double _symmetry_default(self, ImplicitRKTable other) noexcept nogil
    cdef bint _is_symmetric_pair(self, ImplicitRKTable other, double tol) noexcept nogil

    cpdef ImplicitRKTable symplectic_adjoint(self)
    cdef double _symplectic_default(self, ImplicitRKTable other) noexcept nogil
    cpdef bint _is_symplectic_pair(self, ImplicitRKTable other, double tol) noexcept nogil