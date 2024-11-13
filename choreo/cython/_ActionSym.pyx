import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport fabs as cfabs

cimport scipy.linalg.cython_blas

import choreo.scipy_plus.linalg
import networkx

@cython.cdivision(True)
cdef Py_ssize_t gcd (Py_ssize_t a, Py_ssize_t b) noexcept nogil:

    cdef Py_ssize_t c
    while ( a != 0 ):
        c = a
        a = b % a
        b = c

    return b

cdef double default_atol = 1e-10

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

    @property
    def BodyPerm(self):
        return np.asarray(self._BodyPerm)

    @property
    def SpaceRot(self):
        return np.asarray(self._SpaceRot)

    @cython.cdivision(True)
    def __init__(
        self                    ,
        Py_ssize_t[::1] BodyPerm      ,
        double[:,::1] SpaceRot  ,
        Py_ssize_t TimeRev            ,
        Py_ssize_t TimeShiftNum       ,
        Py_ssize_t TimeShiftDen       ,
    ):

        cdef Py_ssize_t den
        cdef Py_ssize_t num

        if TimeShiftDen > 0:
            den = TimeShiftDen
            num = ((TimeShiftNum % den) + den) % den
        else:
            den = - TimeShiftDen
            num = (((-TimeShiftNum) % den) + den) % den

        if (num == 0):
            den = 1

        cdef Py_ssize_t g = gcd(num,den)
        num = num // g
        den = den // g

        self._BodyPerm = BodyPerm
        self._SpaceRot = SpaceRot
        self.TimeRev = TimeRev
        self.TimeShiftNum = num
        self.TimeShiftDen = den

    @cython.final
    def __str__(self):

        out  = "ActionSym object\n"
        out += f"BodyPerm:\n"
        out += f"{self.BodyPerm}\n"
        out += f"SpaceRot:\n"
        out += f"{self.SpaceRot}\n"
        out += f"TimeRev: {self.TimeRev}\n"
        out += f"TimeShift: {self.TimeShiftNum} / {self.TimeShiftDen}"

        return out

    @cython.final
    def __repr__(self):
        return self.__str__()

    @cython.final
    def __format__(self, format_spec):
        return self.__str__()
    
    @cython.final
    @staticmethod
    def Identity(Py_ssize_t nbody, Py_ssize_t geodim):
        """
        Identity: Returns the identity transformation
        """        

        return ActionSym(
            BodyPerm  = np.array(range(nbody), dtype = np.intp) ,
            SpaceRot  = np.identity(geodim, dtype = np.float64) ,
            TimeRev   = 1                                       ,
            TimeShiftNum = 0                                    ,
            TimeShiftDen = 1                                    ,
        )

    @cython.final
    @staticmethod
    def Random(Py_ssize_t nbody, Py_ssize_t geodim, Py_ssize_t maxden = -1):
        """
        Random Returns a random transformation
        """        

        if maxden < 0:
            maxden = 10*nbody

        perm = np.random.permutation(nbody).astype(np.intp)

        rotmat = np.ascontiguousarray(choreo.scipy_plus.linalg.random_orthogonal_matrix(geodim))

        timerev = 1 if np.random.random_sample() < 0.5 else -1

        den = np.random.randint(low = 1, high = maxden)
        num = np.random.randint(low = 0, high =    den)

        return ActionSym(
            BodyPerm = perm     ,
            SpaceRot = rotmat   ,
            TimeRev = timerev   ,
            TimeShiftNum = num  ,
            TimeShiftDen = den  ,
        )

    @cython.final
    cpdef ActionSym Inverse(ActionSym self):
        r"""
        Returns the inverse of a symmetry transformation
        """

        cdef Py_ssize_t[::1] InvPerm = np.zeros(self._BodyPerm.shape[0], dtype = np.intp)
        cdef Py_ssize_t ib
        for ib in range(self._BodyPerm.shape[0]):
            InvPerm[self._BodyPerm[ib]] = ib

        return ActionSym(
            BodyPerm = InvPerm                                  ,
            SpaceRot = self._SpaceRot.T.copy()                  ,
            TimeRev = self.TimeRev                              ,         
            TimeShiftNum = - self.TimeRev * self.TimeShiftNum   ,
            TimeShiftDen = self.TimeShiftDen                    ,
        )

    @cython.final
    cpdef ActionSym TimeDerivative(ActionSym self):
        r"""
        Returns the time derivative of a symmetry transformation.
        If self transforms positions, then self.TimeDerivative() transforms speeds
        """

        cdef double[:, ::1] SpaceRot = self._SpaceRot.copy()
        for i in range(SpaceRot.shape[0]):
            for j in range(SpaceRot.shape[1]):
                SpaceRot[i, j] *= self.TimeRev

        return ActionSym(
            BodyPerm = self._BodyPerm.copy()    ,
            SpaceRot = SpaceRot                 ,
            TimeRev = self.TimeRev              ,         
            TimeShiftNum = self.TimeShiftNum    ,
            TimeShiftDen = self.TimeShiftDen    ,
        )

    @cython.final
    @cython.cdivision(True)
    cpdef ActionSym Compose(ActionSym B, ActionSym A):
        r"""
        Returns the composition of two transformations.

        B.Compose(A) returns the composition B o A, i.e. applies A then B.
        """

        cdef Py_ssize_t[::1] ComposeBodyPerm = np.zeros(B._BodyPerm.shape[0], dtype = np.intp)
        cdef Py_ssize_t ib
        for ib in range(B._BodyPerm.shape[0]):
            ComposeBodyPerm[ib] = B._BodyPerm[A._BodyPerm[ib]]

        cdef Py_ssize_t trev = B.TimeRev * A.TimeRev
        cdef Py_ssize_t num = B.TimeRev * A.TimeShiftNum * B.TimeShiftDen + B.TimeShiftNum * A.TimeShiftDen
        cdef Py_ssize_t den = A.TimeShiftDen * B.TimeShiftDen

        return ActionSym(
            BodyPerm = ComposeBodyPerm                      ,
            SpaceRot = np.matmul(B._SpaceRot,A._SpaceRot)   ,
            TimeRev = trev                                  ,
            TimeShiftNum = num                              ,
            TimeShiftDen = den                              ,
        )

    @cython.final
    cpdef ActionSym Conjugate(ActionSym A, ActionSym B):
        r"""
        Returns the conjugation of a transformation wrt another transformation.

        A.Conjugate(B) returns the conjugation B o A o B^-1.
        """

        return B.Compose(A.Compose(B.Inverse()))

    @cython.final
    cpdef bint IsWellFormed(ActionSym self, double atol = default_atol):
        r"""
        Returns True if the transformation is well-formed.
        """       
        
        cdef bint res = True
        cdef Py_ssize_t i, j

        res = res and (self.TimeShiftNum >= 0)
        res = res and (self.TimeShiftDen >  0)
        res = res and (self.TimeShiftNum <  self.TimeShiftDen)

        for i in range(self._BodyPerm.shape[0]):

            res = res and (self._BodyPerm[i] >= 0) 
            res = res and (self._BodyPerm[i] < self._BodyPerm.shape[0]) 

        unique_perm = np.unique(np.asarray(self._BodyPerm))
        res = res and (unique_perm.shape[0] == self._BodyPerm.shape[0])

        return res

    @cython.final
    cpdef bint IsIdentity(ActionSym self, double atol = default_atol):
        r"""
        Returns True if the transformation is close to identity.
        """       

        return ( 
            self.IsIdentityPerm() and
            self.IsIdentityRot(atol = atol) and
            self.IsIdentityTimeRev() and
            self.IsIdentityTimeShift()
        )

    @cython.final
    cpdef bint IsIdentityPerm(ActionSym self):

        cdef bint isid = True
        cdef Py_ssize_t ib
        cdef Py_ssize_t nbody = self._BodyPerm.shape[0]

        for ib in range(nbody):
            isid = isid and (self._BodyPerm[ib] == ib)

        return isid
    
    @cython.final
    cpdef bint IsIdentityRot(ActionSym self, double atol = default_atol):

        cdef bint isid = True
        cdef Py_ssize_t idim, jdim
        cdef Py_ssize_t geodim = self._SpaceRot.shape[0]

        for idim in range(geodim):
            isid = isid and (cfabs(self._SpaceRot[idim, idim] - 1.) < atol)

        for idim in range(geodim-1):
            for jdim in range(idim+1,geodim):

                isid = isid and (cfabs(self._SpaceRot[idim, jdim]) < atol)
                isid = isid and (cfabs(self._SpaceRot[jdim, idim]) < atol)

        return isid

    @cython.final
    cpdef bint IsIdentityTimeRev(ActionSym self):
        return (self.TimeRev == 1)

    @cython.final    
    cpdef bint IsIdentityTimeShift(ActionSym self):
        return (self.TimeShiftNum == 0)

    @cython.final    
    cpdef bint IsIdentityPermAndRot(ActionSym self, double atol = default_atol):
        return self.IsIdentityPerm() and self.IsIdentityRot(atol = atol)    

    @cython.final    
    cpdef bint IsIdentityPermAndRotAndTimeRev(ActionSym self, double atol = default_atol):
        return self.IsIdentityPerm() and self.IsIdentityRot(atol = atol) and self.IsIdentityTimeRev()

    @cython.final    
    cpdef bint IsIdentityRotAndTimeRev(ActionSym self, double atol = default_atol):
        return self.IsIdentityTimeRev() and self.IsIdentityRot(atol = atol)    

    @cython.final
    cpdef bint IsIdentityRotAndTime(ActionSym self, double atol = default_atol):
        return self.IsIdentityTimeRev() and self.IsIdentityRot(atol = atol) and self.IsIdentityTimeShift()

    @cython.final
    cpdef bint IsSame(ActionSym self, ActionSym other, double atol = default_atol):
        r"""
        Returns True if the two transformations are almost identical.
        """   
        return ((self.Inverse()).Compose(other)).IsIdentity(atol = atol)
    
    @cython.final
    cpdef bint IsSamePerm(ActionSym self, ActionSym other):
        return ((self.Inverse()).Compose(other)).IsIdentityPerm()    

    @cython.final
    cpdef bint IsSameRot(ActionSym self, ActionSym other, double atol = default_atol):

        cdef bint isid = True
        cdef Py_ssize_t idim, jdim
        cdef Py_ssize_t geodim = self._SpaceRot.shape[0]

        for idim in range(geodim):
            for jdim in range(geodim):
                isid = isid and (cfabs(self._SpaceRot[idim, jdim] - other._SpaceRot[idim, jdim]) < atol)

        return isid
    
    @cython.final
    cpdef bint IsSameTimeRev(ActionSym self, ActionSym other):
        return ((self.Inverse()).Compose(other)).IsIdentityTimeRev()    
    
    @cython.final
    cpdef bint IsSameTimeShift(ActionSym self, ActionSym other, double atol = default_atol):
        return ((self.Inverse()).Compose(other)).IsIdentityTimeShift()

    @cython.final    
    cpdef bint IsSamePermAndRot(ActionSym self, ActionSym other, double atol = default_atol):
        return ((self.Inverse()).Compose(other)).IsIdentityPermAndRot(atol = atol)

    @cython.final
    cpdef bint IsSameRotAndTimeRev(ActionSym self, ActionSym other, double atol = default_atol):
        return ((self.Inverse()).Compose(other)).IsIdentityRotAndTimeRev(atol = atol)
    
    @cython.final
    cpdef bint IsSameRotAndTime(ActionSym self, ActionSym other, double atol = default_atol):
        return ((self.Inverse()).Compose(other)).IsIdentityRotAndTime(atol = atol)

    @cython.final
    @cython.cdivision(True)
    cpdef (Py_ssize_t, Py_ssize_t) ApplyTInv(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden):

        cdef Py_ssize_t num = self.TimeRev * (tnum * self.TimeShiftDen - self.TimeShiftNum * tden)
        cdef Py_ssize_t den = tden * self.TimeShiftDen
        num = ((num % den) + den) % den

        cdef Py_ssize_t g = gcd(num,den)
        num = num // g
        den = den // g

        return  num, den

    @cython.final
    @cython.cdivision(True)
    cpdef (Py_ssize_t, Py_ssize_t) ApplyTInvSegm(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden):

        if (self.TimeRev == -1): 
            tnum = ((tnum+1)%tden)

        return self.ApplyTInv(tnum, tden)

    @cython.final
    @cython.cdivision(True)
    cpdef (Py_ssize_t, Py_ssize_t) ApplyT(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden):

        cdef Py_ssize_t num = self.TimeRev * tnum * self.TimeShiftDen + self.TimeShiftNum * tden
        cdef Py_ssize_t den = tden * self.TimeShiftDen
        
        num = ((num % den) + den) % den
        cdef Py_ssize_t g = gcd(num,den)
        num = num // g
        den = den // g

        return  num, den

    @cython.final
    @cython.cdivision(True)
    cpdef (Py_ssize_t, Py_ssize_t) ApplyTSegm(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden):

        if (self.TimeRev == -1): 
            tnum = ((tnum+1)%tden)

        return self.ApplyT(tnum, tden)

    # THIS IS NOT A PERFORMANCE-ORIENTED METHOD
    @cython.final
    def TransformPos(ActionSym self, in_segm, out):
        np.matmul(in_segm, self._SpaceRot.T, out=out)
            
    # THIS IS NOT A PERFORMANCE-ORIENTED METHOD
    @cython.final
    def TransformSegment(ActionSym self, in_segm, out):

        np.matmul(in_segm, self._SpaceRot.T, out=out)
        if self.TimeRev == -1:
            out[:,:] = out[::-1,:]



