import numpy as np
cimport numpy as np
np.import_array()
 
import choreo.scipy_plus.linalg

class ActionSym():
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

    def __init__(
        self,
        BodyPerm ,
        SpaceRot ,
        TimeRev  ,
        TimeShiftNum,
        TimeShiftDen,
    ):

        num = ((TimeShiftNum % TimeShiftDen) + TimeShiftDen) % TimeShiftDen

        if (num == 0):
            den = 1
        else:
            den = TimeShiftDen

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
    def Identity(nbody, geodim):
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
    def Random(nbody, geodim, maxden = None):
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

        if maxden is None:
            maxden = 10*nbody

        perm = np.random.permutation(nbody)

        rotmat = choreo.scipy_plus.linalg.random_orthogonal_matrix(geodim)

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

    def Inverse(self):
        r"""
        Returns the inverse of a symmetry transformation
        """

        InvPerm = np.empty_like(self.BodyPerm)
        for ib in range(self.BodyPerm.size):
            InvPerm[self.BodyPerm[ib]] = ib

        return ActionSym(
            BodyPerm = InvPerm,
            SpaceRot = self.SpaceRot.T.copy(),
            TimeRev = self.TimeRev,         
            TimeShiftNum = - self.TimeRev * self.TimeShiftNum,
            TimeShiftDen = self.TimeShiftDen
        )

    def Compose(B, A):
        r"""
        Returns the composition of two transformations.

        B.Compose(A) returns the composition B o A, i.e. applies A then B.
        """

        ComposeBodyPerm = np.empty_like(B.BodyPerm)
        for ib in range(B.BodyPerm.size):
            ComposeBodyPerm[ib] = B.BodyPerm[A.BodyPerm[ib]]

        return ActionSym(
            BodyPerm = ComposeBodyPerm,
            SpaceRot = np.matmul(B.SpaceRot,A.SpaceRot),
            TimeRev = (B.TimeRev * A.TimeRev),
            TimeShiftNum = A.TimeRev * B.TimeShiftNum * A.TimeShiftDen + A.TimeShiftNum * B.TimeShiftDen,
            TimeShiftDen = A.TimeShiftDen * B.TimeShiftDen
        )

    def IsIdentity(self, atol = 1e-10):
        r"""
        Returns True if the transformation is close to identity.
        """       

        return ( 
            self.IsIdentityPerm() and
            self.IsIdentityRot(atol = atol) and
            self.IsIdentityTimeRev() and
            self.IsIdentityTimeShift()
        )

    def IsIdentityPerm(self):
        return np.array_equal(self.BodyPerm, np.array(range(self.BodyPerm.size), dtype = np.int_))
    
    def IsIdentityRot(self, atol = 1e-10):
        return np.allclose(
            self.SpaceRot,
            np.identity(self.SpaceRot.shape[0], dtype = np.float64),
            rtol = 0.,
            atol = atol
        )    

    def IsIdentityTimeRev(self):
        return (self.TimeRev == 1)
    
    def IsIdentityTimeShift(self):
        return (self.TimeShiftNum == 0)
    
    def IsIdentityRotAndTimeRev(self, atol = 1e-10):
        return self.IsIdentityTimeRev() and self.IsIdentityRot(atol = atol)    
    
    def IsIdentityRotAndTime(self, atol = 1e-10):
        return self.IsIdentityTimeRev() and self.IsIdentityRot(atol = atol) and self.IsIdentityTimeShift()

    def IsSame(self, other, atol = 1e-10):
        r"""
        Returns True if the two transformations are almost identical.
        """   
        return ((self.Inverse()).Compose(other)).IsIdentity(atol = atol)
    
    def IsSamePerm(self, other):
        return ((self.Inverse()).Compose(other)).IsIdentityPerm()    
    
    def IsSameRot(self, other, atol = 1e-10):
        return ().IsIdentityRot(atol = atol)    
    
    def IsSameTimeRev(self, other):
        return ((self.Inverse()).Compose(other)).IsIdentityTimeRev()    
    
    def IsSameTimeShift(self, other, atol = 1e-10):
        return ((self.Inverse()).Compose(other)).IsIdentityTimeShift()

    def IsSameRotAndTimeRev(self, other, atol = 1e-10):
        return ((self.Inverse()).Compose(other)).IsIdentityRotAndTimeRev(atol = atol)
    
    def IsSameRotAndTime(self, other, atol = 1e-10):
        return ((self.Inverse()).Compose(other)).IsIdentityRotAndTime(atol = atol)

    def ApplyT(self, tnum, tden):

        num = self.TimeRev * (tnum * self.TimeShiftDen - self.TimeShiftNum * tden)
        den = tden * self.TimeShiftDen
        num = ((num % den) + den) % den

        return  num, den
