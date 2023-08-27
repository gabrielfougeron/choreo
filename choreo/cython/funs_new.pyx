import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
 
import choreo.scipy_plus.linalg

@cython.cdivision(True)
cdef inline long gcd (long a, long b) nogil:

    cdef long c
    while ( a != 0 ):
        c = a
        a = b % a
        b = c

    return b

cdef class ActionSym():
    r"""
    This class defines the symmetries of the action
    Useful to detect loops and constraints.

    Syntax : Giving one ActionSym to setup_changevar prescribes the following symmetry / constraint :

    .. math::
        x_{\text{LoopTarget}}(t) = \text{SpaceRot} \cdot x_{\text{LoopSource}} (\text{TimeRev} * (t - \text{TimeShift}))

    Where SpaceRot is assumed orthogonal (never actually checked, so beware)
    and TimeShift is defined as a rational fraction.

    cf Palais' principle of symmetric criticality
    """

    cdef public long[::1] BodyPerm
    cdef public double[:,::1] SpaceRot
    cdef public long TimeRev
    cdef public long TimeShiftNum
    cdef public long TimeShiftDen

    @cython.cdivision(True)
    def __init__(
        self,
        long[::1] BodyPerm,
        double[:,::1] SpaceRot,
        long TimeRev,
        long TimeShiftNum,
        long TimeShiftDen,
    ):

        cdef long num = ((TimeShiftNum % TimeShiftDen) + TimeShiftDen) % TimeShiftDen
        cdef long den

        if (num == 0):
            den = 1
        else:
            den = TimeShiftDen

        cdef long g = gcd(num,den)
        num = num // g
        den = den // g

        self.BodyPerm = BodyPerm
        self.SpaceRot = SpaceRot
        self.TimeRev = TimeRev
        self.TimeShiftNum = num
        self.TimeShiftDen = den

    def __str__(self):

        out  = ""
        out += f"BodyPerm: {self.BodyPerm}\n"
        out += f"SpaceRot: {self.SpaceRot}\n"
        out += f"TimeRev: {self.TimeRev}\n"
        out += f"TimeShift: {self.TimeShiftNum / self.TimeShiftDen}"

        return out
    
    @staticmethod
    def Identity(long nbody, long geodim):
        """Identity: Returns the identity transformation

        :param nbody: Number of bodies
        :type nbody: int
        :param geodim: Dimension of ambient space
        :type geodim: int
        :return: The identity transformation of appropriate size
        :rtype: ActionSym
        """        

        return ActionSym(
            BodyPerm  = np.array(range(nbody), dtype = np.int_),
            SpaceRot  = np.identity(geodim, dtype = np.float64),
            TimeRev   = 1,
            TimeShiftNum = 0,
            TimeShiftDen = 1
        )

    @staticmethod
    def Random(long nbody, long geodim, long maxden = -1):
        """Random Returns a random transformation

        :param nbody: Number of bodies
        :type nbody: int
        :param geodim: Dimension of ambient space
        :type geodim: int
        :param maxden: Maximum denominator for TimeShift, defaults to None
        :type maxden: int, optional
        :return: A random transformation of appropriate size
        :rtype: ActionSym
        """        

        if maxden < 0:
            maxden = 10*nbody

        perm = np.random.permutation(nbody)

        rotmat = np.ascontiguousarray(choreo.scipy_plus.linalg.random_orthogonal_matrix(geodim))

        timerev = 1 if np.random.random_sample() < 0.5 else -1

        den = np.random.randint(low = 1, high = maxden)
        num = np.random.randint(low = 0, high =    den)

        return ActionSym(
            BodyPerm = perm,
            SpaceRot = rotmat,
            TimeRev = timerev,
            TimeShiftNum = num,
            TimeShiftDen = den,
        )

    cpdef ActionSym Inverse(ActionSym self):
        r"""
        Returns the inverse of a symmetry transformation
        """

        cdef long[::1] InvPerm = np.zeros(self.BodyPerm.shape[0], dtype = np.intp)
        cdef long ib
        for ib in range(self.BodyPerm.shape[0]):
            InvPerm[self.BodyPerm[ib]] = ib

        return ActionSym(
            BodyPerm = InvPerm,
            SpaceRot = self.SpaceRot.T.copy(),
            TimeRev = self.TimeRev,         
            TimeShiftNum = - self.TimeRev * self.TimeShiftNum,
            TimeShiftDen = self.TimeShiftDen
        )

    cpdef ActionSym Compose(ActionSym B, ActionSym A):
        r"""
        Returns the composition of two transformations.

        B.Compose(A) returns the composition B o A, i.e. applies A then B.
        """

        cdef long[::1] ComposeBodyPerm = np.zeros(B.BodyPerm.shape[0], dtype = np.intp)
        cdef long ib
        for ib in range(B.BodyPerm.shape[0]):
            ComposeBodyPerm[ib] = B.BodyPerm[A.BodyPerm[ib]]

        cdef long trev = B.TimeRev * A.TimeRev
        cdef long num = A.TimeRev * B.TimeShiftNum * A.TimeShiftDen + A.TimeShiftNum * B.TimeShiftDen
        cdef long den = A.TimeShiftDen * B.TimeShiftDen

        return ActionSym(
            BodyPerm = ComposeBodyPerm,
            SpaceRot = np.matmul(B.SpaceRot,A.SpaceRot),
            TimeRev = trev,
            TimeShiftNum = num,
            TimeShiftDen = den
        )

    cpdef bint IsIdentity(ActionSym self, double atol = 1e-10):
        r"""
        Returns True if the transformation is close to identity.
        """       

        return ( 
            self.IsIdentityPerm() and
            self.IsIdentityRot(atol = atol) and
            self.IsIdentityTimeRev() and
            self.IsIdentityTimeShift()
        )

    cpdef bint IsIdentityPerm(ActionSym self):

        cdef bint isid = True
        cdef long ib
        cdef long nbody = self.BodyPerm.shape[0]

        for ib in range(nbody):
            isid = isid and (self.BodyPerm[ib] == ib)

        return isid
    
    cpdef bint IsIdentityRot(ActionSym self, double atol = 1e-10):

        cdef bint isid = True
        cdef long idim, jdim
        cdef long geodim = self.SpaceRot.shape[0]

        for idim in range(geodim):
            isid = isid and (cfabs(self.SpaceRot[idim, idim] - 1.) < atol)

        for idim in range(geodim-1):
            for jdim in range(idim+1,geodim):

                isid = isid and (cfabs(self.SpaceRot[idim, jdim]) < atol)
                isid = isid and (cfabs(self.SpaceRot[jdim, idim]) < atol)

        return isid

    cpdef bint IsIdentityTimeRev(ActionSym self):
        return (self.TimeRev == 1)
    
    cpdef bint IsIdentityTimeShift(ActionSym self):
        return (self.TimeShiftNum == 0)
    
    cpdef bint IsIdentityRotAndTimeRev(ActionSym self, double atol = 1e-10):
        return self.IsIdentityTimeRev() and self.IsIdentityRot(atol = atol)    
    
    cpdef bint IsIdentityRotAndTime(ActionSym self, double atol = 1e-10):
        return self.IsIdentityTimeRev() and self.IsIdentityRot(atol = atol) and self.IsIdentityTimeShift()

    cpdef bint IsSame(ActionSym self, other, double atol = 1e-10):
        r"""
        Returns True if the two transformations are almost identical.
        """   
        return ((self.Inverse()).Compose(other)).IsIdentity(atol = atol)
    
    cpdef bint IsSamePerm(ActionSym self, other):
        return ((self.Inverse()).Compose(other)).IsIdentityPerm()    
    
    cpdef bint IsSameRot(ActionSym self, other, double atol = 1e-10):
        return ((self.Inverse()).Compose(other)).IsIdentityRot(atol = atol)    
    
    cpdef bint IsSameTimeRev(ActionSym self, ActionSym other):
        return ((self.Inverse()).Compose(other)).IsIdentityTimeRev()    
    
    cpdef bint IsSameTimeShift(ActionSym self, ActionSym other, double atol = 1e-10):
        return ((self.Inverse()).Compose(other)).IsIdentityTimeShift()

    cpdef bint IsSameRotAndTimeRev(ActionSym self, ActionSym other, double atol = 1e-10):
        return ((self.Inverse()).Compose(other)).IsIdentityRotAndTimeRev(atol = atol)
    
    cpdef bint IsSameRotAndTime(ActionSym self, ActionSym other, double atol = 1e-10):
        return ((self.Inverse()).Compose(other)).IsIdentityRotAndTime(atol = atol)

    @cython.cdivision(True)
    cpdef (long, long)ApplyT(ActionSym self, long tnum, long tden):

        cdef long num = self.TimeRev * (tnum * self.TimeShiftDen - self.TimeShiftNum * tden)
        cdef long den = tden * self.TimeShiftDen
        num = ((num % den) + den) % den

        cdef long g = gcd(num,den)
        num = num // g
        den = den // g

        return  num, den