cimport numpy as np
cimport cython

@cython.final
cdef class ActionSym():

    cdef Py_ssize_t[::1] _BodyPerm
    cdef double[:,::1] _SpaceRot
    cdef readonly Py_ssize_t TimeRev
    """:class:`python:int` Time reversal
    
    A value of ``-1`` denotes time reversal, and a value of ``1`` denotes no time reversal.
    """
    cdef readonly Py_ssize_t TimeShiftNum
    """:class:`python:int` Numerator of the rational time shift.
    """
    cdef readonly Py_ssize_t TimeShiftDen
    """:class:`python:int` Denominator of the rational time shift.
    """

    cpdef ActionSym Inverse(ActionSym self)

    cpdef ActionSym TimeDerivative(ActionSym self)

    cpdef ActionSym Compose(ActionSym B, ActionSym A)

    cpdef ActionSym Conjugate(ActionSym A, ActionSym B)

    cpdef bint IsWellFormed(ActionSym self, double atol = *)

    cpdef bint IsIdentity(ActionSym self, double atol = *)

    cpdef bint IsIdentityPerm(ActionSym self)

    cpdef bint IsIdentityRot(ActionSym self, double atol = *)

    cpdef bint IsIdentityTimeRev(ActionSym self)

    cpdef bint IsIdentityTimeShift(ActionSym self)
 
    cpdef bint IsIdentityPermAndRot(ActionSym self, double atol = *)

    cpdef bint IsIdentityPermAndRotAndTimeRev(ActionSym self, double atol = *)

    cpdef bint IsIdentityRotAndTimeRev(ActionSym self, double atol = *)

    cpdef bint IsIdentityRotAndTime(ActionSym self, double atol = *)

    cpdef bint IsSame(ActionSym self, ActionSym other, double atol = *)
    
    cpdef bint IsSamePerm(ActionSym self, ActionSym other) 

    cpdef bint IsSameRot(ActionSym self, ActionSym other, double atol = *)
    
    cpdef bint IsSameTimeRev(ActionSym self, ActionSym other) 
    
    cpdef bint IsSameTimeShift(ActionSym self, ActionSym other, double atol = *)

    cpdef bint IsSameRotAndTimeRev(ActionSym self, ActionSym other, double atol = *)
    
    cpdef bint IsSameRotAndTime(ActionSym self, ActionSym other, double atol = *)

    cpdef (Py_ssize_t, Py_ssize_t) ApplyTInv(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden)

    cpdef (Py_ssize_t, Py_ssize_t) ApplyTInvSegm(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden)

    cpdef (Py_ssize_t, Py_ssize_t) ApplyT(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden)

    cpdef (Py_ssize_t, Py_ssize_t) ApplyTSegm(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden)


