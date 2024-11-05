cimport numpy as np
cimport cython

@cython.final
cdef class ActionSym():
    """
    This class defines the symmetries of the action
    Useful to detect loops and constraints.

    Syntax : Giving one ActionSym to setup_changevar prescribes the following symmetry / constraint :

    .. math::
        x_{\\text{LoopTarget}}(t) = \\text{SpaceRot} \cdot x_{\\text{LoopSource}} (\\text{TimeRev} * (t - \\text{TimeShift}))

    Where SpaceRot is assumed orthogonal (never actually checked, so beware)
    and TimeShift is defined as a rational fraction.

    cf Palais' principle of symmetric criticality
    """

    cdef long[::1] _BodyPerm
    cdef double[:,::1] _SpaceRot
    cdef readonly long TimeRev
    cdef readonly long TimeShiftNum
    cdef readonly long TimeShiftDen

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

    cpdef bint IsSamePermAndRot(ActionSym self, ActionSym other, double atol = *)

    cpdef bint IsSameRotAndTimeRev(ActionSym self, ActionSym other, double atol = *)
    
    cpdef bint IsSameRotAndTime(ActionSym self, ActionSym other, double atol = *)

    cpdef (long, long) ApplyTInv(ActionSym self, long tnum, long tden)

    cpdef (long, long) ApplyTInvSegm(ActionSym self, long tnum, long tden)

    cpdef (long, long) ApplyT(ActionSym self, long tnum, long tden)

    cpdef (long, long) ApplyTSegm(ActionSym self, long tnum, long tden)


