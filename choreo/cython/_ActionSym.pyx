import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport fabs as cfabs
from libc.math cimport sqrt as csqrt
from libc.math cimport floor as cfloor

cimport scipy.linalg.cython_blas
cimport scipy.linalg.cython_lapack
from choreo.scipy_plus.cython.blas_consts cimport *

import choreo.scipy_plus.linalg
import networkx
import itertools
import string

@cython.cdivision(True)
cdef Py_ssize_t gcd (Py_ssize_t a, Py_ssize_t b) noexcept nogil:

    cdef Py_ssize_t c
    while ( a != 0 ):
        c = a
        a = b % a
        b = c

    return b

cdef double default_atol = 1e-10

@cython.auto_pickle(False)
@cython.final
cdef class ActionSym():
    r"""This class defines the symmetries in a N-body system.

    A symmetry :math:`\sigma` is a transformation of paths that leaves the physics of a N-body system invariant.

    .. math::
        x_j(t) = \mathrm{R} \cdot x_i (s \cdot (t - \Delta t))

    where:
    
    * :math:`x_i` and :math:`x_j` where :math:`j =`:attr:`BodyPerm`:math:`(i)`  are the positions in the source and target loops, respectively.
    * :math:`\mathrm{R}` is an orthogonal matrix corresponding to :attr:`SpaceRot`.
    * :math:`s` corresponds to :attr:`TimeRev` is either ``1`` or ``-1`` and denotes whether time flows forwards or backwards.
    * :math:`\Delta t` denotes a **rationnal** time shift of the form :attr:`TimeShiftNum` / :attr:`TimeShiftDen` .

    Useful to detect loops and constraints.

    .. todo:: link to better explanation whend done
        cf Palais' principle of symmetric criticality in :footcite:`palais1979principle`
    
    :cited:
    .. footbibliography::

    """

    @property
    def BodyPerm(self):
        """:class:`numpy:numpy.ndarray`:class:`(shape = (nbody), dtype = np.intp)` Permutation of the bodies.
        """
        return np.asarray(self._BodyPerm)

    @property
    def SpaceRot(self):
        """ :class:`numpy:numpy.ndarray`:class:`(shape = (nbody), dtype = np.intp)` Isometry of space.
        """
        return np.asarray(self._SpaceRot)

    @cython.cdivision(True)
    def __init__(
        self                        ,
        Py_ssize_t[::1] BodyPerm    ,
        double[:,::1] SpaceRot      ,
        Py_ssize_t TimeRev          ,
        Py_ssize_t TimeShiftNum     ,
        Py_ssize_t TimeShiftDen     ,
    ):
        """Defines a symmetry of the action functional.

        Parameters
        ----------
        BodyPerm : :class:`numpy:numpy.ndarray`:class:`(shape = (nbody), dtype = np.intp)`
            Permutation of the bodies.
        SpaceRot : :class:`numpy:numpy.ndarray`:class:`(shape = (geodim, geodim), dtype = np.float64)`
            Isometry of space. This matrix is assumed orthogonal, which is not automatically checked (cf :meth:`IsWellFormed`).
        TimeRev : :class:`python:int`
            A value of ``-1`` denotes time reversal, and a value of ``1`` denotes no time reversal.
        TimeShiftNum : :class:`python:int`
            Numerator of the rational time shift.
        TimeShiftDen : :class:`python:int`
            Denominator of the rational time shift.
        """    

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
    def FromDict(Sym_dict):
        """Returns a new :class:`choreo.ActionSym` from a :class:`python:dict`.

        This is a helper function designed to facilitate loading symmetries from configuration and solution files.

        Parameters
        ----------
        Sym_dict : :class:`python:dict` 

        Returns
        -------
        :class:`choreo.ActionSym`

        """

        TimeRev = Sym_dict["TimeRev"]

        if isinstance(TimeRev, str):
            if (TimeRev == "True"):
                TimeRev = -1
            elif (TimeRev == "False"):
                TimeRev = 1
            else:
                raise ValueError('TimeRev given as a string must be "True" or "False"')

        return ActionSym(
            np.array(Sym_dict["BodyPerm"], dtype=np.intp   )    ,
            np.array(Sym_dict["SpaceRot"], dtype=np.float64)    ,
            TimeRev                                             ,
            Sym_dict["TimeShiftNum"]                            ,
            Sym_dict["TimeShiftDen"]                            ,
        )

    @cython.final
    @staticmethod
    def Identity(Py_ssize_t nbody, Py_ssize_t geodim):
        """Returns the identity transformation.

        Example
        -------

        >>> import choreo
        >>> nbody = 10
        >>> geodim = 4
        >>> choreo.ActionSym.Identity(nbody, geodim).IsIdentity()
        True
        >>> print(choreo.ActionSym.Identity(nbody, geodim))
        ActionSym object
        BodyPerm:
        [0 1 2 3 4 5 6 7 8 9]
        SpaceRot:
        [[1. 0. 0. 0.]
        [0. 1. 0. 0.]
        [0. 0. 1. 0.]
        [0. 0. 0. 1.]]
        TimeRev: 1
        TimeShift: 0 / 1

        Parameters
        ----------
        nbody : :class:`python:int`
            Number of bodies in the system.
        geodim : :class:`python:int`
            Dimension of ambiant space.

        Returns
        -------
        :class:`choreo.ActionSym`
            An identity transformation of a system of ``nbody`` point masses in dimension ``geodim``.
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
        """Returns a random transformation.

        .. warning :: The underlying probability density is **NOT** uniform.

        Parameters
        ----------
        nbody : :class:`python:int`
            Number of bodies in the system.
        geodim : :class:`python:int`
            Dimension of ambiant space.
        maxden : :class:`python:int`, optional
            Maximum denominator of the time shift. Negative values will be given a maximum denominator equal to ``10 * nbody``.\n
            By default -1

        Returns
        -------
        :class:`choreo.ActionSym`
            A random transformation of a system of ``nbody`` point masses in dimension ``geodim``.
        """          

        if maxden < 0:
            maxden = 10*nbody

        perm = np.random.permutation(nbody).astype(np.intp)

        rotmat = ActionSym.SurjectiveDirectSpaceRot(np.random.random(geodim*(geodim-1)//2))
        if np.random.random_sample() < 0.5:
            rotmat[0,:] *= -1

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
        """ Returns the inverse of a symmetry transformation.

        For all well-formed transformation  ``A``, the inverse transformation ``A.Inverse()`` satisfies ``A.Inverse().Compose(A).IsIdentity() is True``.

        Example
        -------

        >>> import choreo
        >>> nbody = 10
        >>> geodim = 4
        >>> A = choreo.ActionSym.Random(nbody, geodim)
        >>> A.Inverse().Compose(A).IsIdentity()
        True

        Returns
        -------
        :class:`choreo.ActionSym`
            The inverse transformation.
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
        """ Returns the time derivative of a symmetry transformation.

        If ``A`` transforms positions, then ``A.TimeDerivative()`` transforms velocities.

        Returns
        -------
        :class:`choreo.ActionSym`
            The time derivative of the input transformation

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
        """ Returns the composition of two transformations.

        ``B.Compose(A)`` returns the composition :math:`B \circ A`, i.e. applies ``A`` then ``B``.

        Parameters
        ----------
        B, A : :class:`choreo.ActionSym`
            Input transformation.

        Returns
        -------
        :class:`choreo.ActionSym`
            The composition of the input transformations.
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
        """Returns the conjugation of a transformation with respect to another transformation.

        ``A.Conjugate(B)`` returns the conjugation :math:`B \circ A \circ B^{-1}`.
        
        Parameters
        ----------
        A, B: :class:`choreo.ActionSym`
            Input transformation.

        Returns
        -------
        :class:`choreo.ActionSym`
            The conjugation of the input transformations.

        """

        return B.Compose(A.Compose(B.Inverse()))

    @cython.final
    cpdef bint IsWellFormed(ActionSym self, double atol = default_atol):
        """Returns :data:`python:True` if the transformation is well-formed.

        This function will return :data:`python:True` if and only if **all** the following constraints are satisfied:

        * TimeShift = :attr:`TimeShiftNum` / :attr:`TimeShiftDen`  is an irreducible fraction in :math:`[0,1[`.
        * :attr:`BodyPerm` defines a permutation of [0, ..., n-1], where n = :attr:`BodyPerm`.shape(0).
        * :attr:`SpaceRot` is an orthogonal matrix.
        
        Parameters
        ----------
        atol : :class:`python:float`, optional
            Absolute tolerance for the orthogonality test.

        Returns
        -------
        :class:`python:bool`

        """       
        
        cdef bint res = True
        cdef Py_ssize_t i, j,k
        cdef double dot

        res = res and (self.TimeShiftNum >= 0)
        res = res and (self.TimeShiftDen >  0)
        res = res and (self.TimeShiftNum <  self.TimeShiftDen)

        for i in range(self._BodyPerm.shape[0]):

            res = res and (self._BodyPerm[i] >= 0) 
            res = res and (self._BodyPerm[i] < self._BodyPerm.shape[0]) 

        unique_perm = np.unique(np.asarray(self._BodyPerm))
        res = res and (unique_perm.shape[0] == self._BodyPerm.shape[0])

        res = res and (self._SpaceRot.shape[0] == self._SpaceRot.shape[1])

        for i in range(self._SpaceRot.shape[0]):
            for j in range(self._SpaceRot.shape[0]):
                dot = 0.
                for k in range(self._SpaceRot.shape[0]):
                    dot += self._SpaceRot[i,k] *  self._SpaceRot[j,k]

                if i == j :
                    res = res and (cfabs(dot - 1.) < atol)
                else:
                    res = res and (cfabs(dot) < atol)

        return res

    @cython.final
    cpdef bint IsIdentity(ActionSym self, double atol = default_atol):
        """Returns :data:`python:True` if the transformation is within ``atol`` of the identity.

        Example
        -------

        >>> import choreo
        >>> nbody = 10
        >>> geodim = 4
        >>> choreo.ActionSym.Identity(nbody, geodim).IsIdentity()
        True

        Parameters
        ----------
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsIdentityPerm`
        * :meth:`choreo.ActionSym.IsIdentityRot`
        * :meth:`choreo.ActionSym.IsIdentityTimeRev`
        * :meth:`choreo.ActionSym.IsIdentityTimeShift`
        
        """    

        return ( 
            self.IsIdentityPerm() and
            self.IsIdentityRot(atol = atol) and
            self.IsIdentityTimeRev() and
            self.IsIdentityTimeShift()
        )

    @cython.final
    cpdef bint IsIdentityPerm(ActionSym self):
        """Returns :data:`python:True` if the body permutation part of the transformation is the identity permutation.

        Returns
        -------
        :class:`python:bool`
        
        """    

        cdef bint isid = True
        cdef Py_ssize_t ib
        cdef Py_ssize_t nbody = self._BodyPerm.shape[0]

        for ib in range(nbody):
            isid = isid and (self._BodyPerm[ib] == ib)

        return isid
    
    @cython.final
    cpdef bint IsIdentityRot(ActionSym self, double atol = default_atol):
        """Returns :data:`python:True` if the space isometry part of the transformation is within ``atol`` of the identity.
        
        Parameters
        ----------
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`
        
        """    

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
        """Returns :data:`python:True` if the transformation does not reverse time.

        Returns
        -------
        :class:`python:bool`
        
        """    
        return (self.TimeRev == 1)

    @cython.final    
    cpdef bint IsIdentityTimeShift(ActionSym self):
        """Returns :data:`python:True` if the transformation does not shift time.

        Returns
        -------
        :class:`python:bool`
        
        """    
        return (self.TimeShiftNum == 0)

    @cython.final    
    cpdef bint IsIdentityPermAndRot(ActionSym self, double atol = default_atol):
        """ Returns :data:`python:True` if parts of the transformation are within ``atol`` of the identity.
        
        Returns :data:`python:True` if:
        
        * The body permutation part of the transformation is the identity permutation.
        * The space isometry part of the transformation is within ``atol`` of the identity.

        Parameters
        ----------
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsIdentityPerm`
        * :meth:`choreo.ActionSym.IsIdentityRot`

        """    
        return self.IsIdentityPerm() and self.IsIdentityRot(atol = atol)    

    @cython.final    
    cpdef bint IsIdentityPermAndRotAndTimeRev(ActionSym self, double atol = default_atol):
        """ Returns :data:`python:True` if parts of the transformation are within ``atol`` of the identity.
        
        Returns :data:`python:True` if:
        
        * The body permutation part of the transformation is the identity permutation.
        * The space isometry part of the transformation is within ``atol`` of the identity.
        * The transformation does not reverse time.

        Parameters
        ----------
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsIdentityPerm`
        * :meth:`choreo.ActionSym.IsIdentityRot`
        * :meth:`choreo.ActionSym.IsIdentityTimeRev`

        """  
        return self.IsIdentityPerm() and self.IsIdentityRot(atol = atol) and self.IsIdentityTimeRev()

    @cython.final    
    cpdef bint IsIdentityRotAndTimeRev(ActionSym self, double atol = default_atol):
        """ Returns :data:`python:True` if parts of the transformation are within ``atol`` of the identity.
        
        Returns :data:`python:True` if:

        * The space isometry part of the transformation is within ``atol`` of the identity.
        * The transformation does not reverse time.

        Parameters
        ----------
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsIdentityRot`
        * :meth:`choreo.ActionSym.IsIdentityTimeRev`

        """  
        return self.IsIdentityTimeRev() and self.IsIdentityRot(atol = atol)    

    @cython.final
    cpdef bint IsIdentityRotAndTime(ActionSym self, double atol = default_atol):
        """ Returns :data:`python:True` if parts of the transformation are within ``atol`` of the identity.
        
        Returns :data:`python:True` if:
        
        * The space isometry part of the transformation is within ``atol`` of the identity.
        * The transformation does not shift or reverse time.

        Parameters
        ----------
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsIdentityRot`
        * :meth:`choreo.ActionSym.IsIdentityTimeRev`
        * :meth:`choreo.ActionSym.IsIdentityTimeShift`

        """  
        return self.IsIdentityTimeRev() and self.IsIdentityTimeShift() and self.IsIdentityRot(atol = atol) 

    @cython.final
    cpdef bint IsSame(ActionSym self, ActionSym other, double atol = default_atol):
        """Returns :data:`python:True` if the two transformations are within ``atol`` of each other.

        Parameters
        ----------
        other : :class:`choreo.ActionSym`
            Input transformation.
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsSamePerm`
        * :meth:`choreo.ActionSym.IsSameRot`
        * :meth:`choreo.ActionSym.IsSameTimeRev`
        * :meth:`choreo.ActionSym.IsSameTimeShift`

        """   
        return ( 
            self.IsSamePerm(other) and
            self.IsSameRot(other, atol = atol) and
            self.IsSameTimeRev(other) and
            self.IsSameTimeShift(other)
        )
    
    @cython.final
    cpdef bint IsSamePerm(ActionSym self, ActionSym other):
        """Returns :data:`python:True` if the two transformations have identical body permutations.

        Parameters
        ----------
        other : :class:`choreo.ActionSym`
            Input transformation.

        Returns
        -------
        :class:`python:bool`

        """   

        cdef bint isid = True
        cdef Py_ssize_t ib
        cdef Py_ssize_t nbody = self._BodyPerm.shape[0]

        for ib in range(nbody):
            isid = isid and (self._BodyPerm[ib] == other._BodyPerm[ib])

        return isid  

    @cython.final
    cpdef bint IsSameRot(ActionSym self, ActionSym other, double atol = default_atol):
        """Returns :data:`python:True` if the two transformations have space isometries within ``atol`` of each other.

        Parameters
        ----------
        other : :class:`choreo.ActionSym`
            Input transformation.
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        """   

        cdef bint isid = True
        cdef Py_ssize_t idim, jdim
        cdef Py_ssize_t geodim = self._SpaceRot.shape[0]

        for idim in range(geodim):
            for jdim in range(geodim):
                isid = isid and (cfabs(self._SpaceRot[idim, jdim] - other._SpaceRot[idim, jdim]) < atol)

        return isid
    
    @cython.final
    cpdef bint IsSameTimeRev(ActionSym self, ActionSym other):
        """Returns :data:`python:True` if the two transformations have identical time reversal.

        Parameters
        ----------
        other : :class:`choreo.ActionSym`
            Input transformation.

        Returns
        -------
        :class:`python:bool`

        """   
        return self.TimeRev == other.TimeRev
    
    @cython.final
    cpdef bint IsSameTimeShift(ActionSym self, ActionSym other, double atol = default_atol):
        """Returns :data:`python:True` if the two transformations have identical time shifts.

        Parameters
        ----------
        other : :class:`choreo.ActionSym`
            Input transformation.

        Returns
        -------
        :class:`python:bool`

        """   
        return self.TimeShiftNum * other.TimeShiftDen == self.TimeShiftDen * other.TimeShiftNum    

    @cython.final
    cpdef bint IsSameRotAndTimeRev(ActionSym self, ActionSym other, double atol = default_atol):
        """Returns :data:`python:True` if the two transformations have properties within ``atol`` of each other.

        ``A.IsSameRotAndTimeRev(B)`` returns :data:`python:True` if ``A`` and ``B`` have:

        * The space isometry part of the transformation within ``atol`` of each other.
        * Identical time reversal.

        Parameters
        ----------
        other : :class:`choreo.ActionSym`
            Input transformation.
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsSameRot`
        * :meth:`choreo.ActionSym.IsSameTimeRev``

        """   
        return self.IsSameTimeRev(other) and self.IsSameRot(other, atol = atol)    
    
    @cython.final
    cpdef bint IsSameRotAndTime(ActionSym self, ActionSym other, double atol = default_atol):
        """Returns :data:`python:True` if the two transformations have properties within ``atol`` of each other.

        ``A.IsSameRotAndTimeRev(B)`` returns :data:`python:True` if ``A`` and ``B`` have:

        * The space isometry part of the transformation within ``atol`` of each other.
        * Identical time shifts and reversal.

        Parameters
        ----------
        other : :class:`choreo.ActionSym`
            Input transformation.
        atol : :class:`python:float`, optional
            Absolute tolerance.

        Returns
        -------
        :class:`python:bool`

        See Also
        --------

        * :meth:`choreo.ActionSym.IsSameRot`
        * :meth:`choreo.ActionSym.IsSameTimeShift``
        * :meth:`choreo.ActionSym.IsSameTimeRev``

        """   
        return self.IsSameTimeShift(other) and self.IsSameTimeRev(other) and self.IsSameRot(other, atol = atol)    

    @cython.final
    @cython.cdivision(True)
    cpdef (Py_ssize_t, Py_ssize_t) ApplyTInv(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden):
        """Returns a rational inverse transformed time instant given an input rational time instant.

        Parameters
        ----------
        tnum : :class:`python:int`
            Numerator of the input time instant.
        tden : :class:`python:int`
            Denominator of the input time instant.

        Returns
        -------
        :class:`python:int`, :class:`python:int`
            The numerator and denominator of the transformed time instant.

        """    

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
        """Returns a rational inverse transformed time segment given an input rational time segment.

        Parameters
        ----------
        tnum : :class:`python:int`
            Numerator of the input time segment.
        tden : :class:`python:int`
            Denominator of the input time segment.

        Returns
        -------
        :class:`python:int`, :class:`python:int`
            The numerator and denominator of the transformed time segment.

        """ 

        if (self.TimeRev == -1): 
            tnum = ((tnum+1)%tden)

        return self.ApplyTInv(tnum, tden)

    @cython.final
    @cython.cdivision(True)
    cpdef (Py_ssize_t, Py_ssize_t) ApplyT(ActionSym self, Py_ssize_t tnum, Py_ssize_t tden):
        """Returns a rational transformed time instant given an input rational time instant.

        Parameters
        ----------
        tnum : :class:`python:int`
            Numerator of the input time instant.
        tden : :class:`python:int`
            Denominator of the input time instant.

        Returns
        -------
        :class:`python:int`, :class:`python:int`
            The numerator and denominator of the transformed time instant.

        """ 


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
        """Returns a rational transformed time segment given an input rational time segment.

        Parameters
        ----------
        tnum : :class:`python:int`
            Numerator of the input time segment.
        tden : :class:`python:int`
            Denominator of the input time segment.

        Returns
        -------
        :class:`python:int`, :class:`python:int`
            The numerator and denominator of the transformed time segment.

        """ 

        if (self.TimeRev == -1): 
            tnum = ((tnum+1)%tden)

        return self.ApplyT(tnum, tden)

    # THIS IS NOT A PERFORMANCE-ORIENTED METHOD
    @cython.final
    def TransformPos(ActionSym self, in_pos, out):
        """Computes a transformed position by the space isometry.

        Parameters
        ----------
        in_pos : :class:`numpy:numpy.ndarray`:class:`(shape = (..., geodim), dtype = np.float64)`
            Input position.
        out : :class:`numpy:numpy.ndarray`:class:`(shape = (..., geodim), dtype = np.float64)`
            Output transformed position.

        """ 
        np.matmul(in_pos, self._SpaceRot.T, out=out)
            
    # THIS IS NOT A PERFORMANCE-ORIENTED METHOD
    @cython.final
    def TransformSegment(ActionSym self, in_segm, out):
        """Computes a transformed segment of positions by the time and space isometry.

        Parameters
        ----------
        in_pos : :class:`numpy:numpy.ndarray`:class:`(shape = (nsegm, geodim), dtype = np.float64)`
            Input segment of positions.
        out : :class:`numpy:numpy.ndarray`:class:`(shape = (nsegm, geodim), dtype = np.float64)`
            Output transformed segment of positions.

        """ 

        np.matmul(in_segm, self._SpaceRot.T, out=out)
        if self.TimeRev == -1:
            out[:,:] = out[::-1,:]          
            
    @cython.final
    def to_dict(ActionSym self):
        """Returns a :class:`python:dict` containing the transformation informations.

        This is a helper function designed to facilitate writing configuration and solution files.

        Returns
        -------
        :class:`python:dict`

        """ 

        return {
            "BodyPerm"      : self.BodyPerm.tolist()    ,
            "SpaceRot"      : self.SpaceRot.tolist()    ,
            "TimeRev"       : self.TimeRev              ,
            "TimeShiftNum"  : self.TimeShiftNum         ,
            "TimeShiftDen"  : self.TimeShiftDen         ,
        }

    @cython.final
    @cython.cdivision(True)
    @staticmethod
    def TimeShifts(Py_ssize_t max_den):
        """Generates all rational fractions in :math:`[0,1[` with denominator lower or equal to ``max_den``.

        The generated rational time shifts are given in reduced form and increasing order.

        Example
        -------

        >>> import choreo
        >>> for a,b in choreo.ActionSym.TimeShifts(5):
        ...     print(a,b)
        ...
        0 1
        1 5
        1 4
        1 3
        2 5
        1 2
        3 5
        2 3
        3 4
        4 5

        Parameters
        ----------
        max_den : :class:`python:int`
            Maximum fraction denominator.
        
        Returns
        -------
        :class:`python:int`, :class:`python:int`
            The fraction numerator and denominator.

        See Also
        --------

        * `Farey sequence on Wikipedia <https://en.wikipedia.org/wiki/Farey_sequence>`_
        
        """ 

        cdef Py_ssize_t num = 0
        cdef Py_ssize_t den = 1
        cdef Py_ssize_t c = 1
        cdef Py_ssize_t d = max_den

        cdef Py_ssize_t k, p, q, g

        yield (num, den)

        while c < d:

            k = (max_den + den) // d

            p = k * c - num
            q = k * d - den

            g = gcd(c,d)
            num = c // g
            den = d // g

            c = p
            d = q

            yield (num, den)

    @cython.final
    @staticmethod
    def InvolutivePermutations(Py_ssize_t n):
        """Generates all involutive permutations of size ``n``.

        The generated rational time shifts are given in reduced form and increasing order.

        Example
        -------

        >>> import choreo
        >>> for p in choreo.ActionSym.InvolutivePermutations(4):
        ...     print(p)
        ...
        [0 1 2 3]
        [0 1 3 2]
        [0 2 1 3]
        [0 3 2 1]
        [1 0 2 3]
        [1 0 3 2]
        [2 1 0 3]
        [2 3 0 1]
        [3 1 2 0]
        [3 2 1 0]

        Parameters
        ----------
        n : :class:`python:int`
            Permutation size
        
        Returns
        -------
        :class:`numpy:numpy.ndarray`:class:`(shape = n, dtype = np.intp)`
        
        """ 
        for p in itertools.permutations(range(n)):
            for i in range(n):
                if p[p[i]] != i:
                    break
            else:
                yield np.array(p, dtype=np.intp)

    @cython.final
    @staticmethod
    def SurjectiveDirectSpaceRot(double[::1] params):
        """Surjective parametrization of direct isometries.

        This function computes a direct isometry :math:`R \in SO(n)` from a set of :math:`\\frac{n(n-1)}{2}` parameters using the squared Cayley transform, ensuring surjectivity:

        .. math::
            R = ((I_n-A)^{-1}(I_n+A))^2

        where :math:`A` denotes the skew-symmetric matrix whose upper triangular entries are given in ``params``.

        Example
        -------

        >>> R = choreo.ActionSym.SurjectiveDirectSpaceRot(np.array([1.,2.,3.]))
        >>> np.linalg.norm(np.matmul(R,R.T) - np.identity(3)) < 1e-14
        np.True_
        >>> abs(np.linalg.det(R) - 1.) < 1e-14
        np.True_

        Parameters
        ----------
        params : :class:`numpy:numpy.ndarray`:class:`(shape = n*(n-1)/2, dtype = np.float64)`
            Upper part of the Cayley skew-symmetric matrix.
        
        Returns
        -------
        :class:`numpy:numpy.ndarray`:class:`(shape = (n,n), dtype = np.float64)`
            A direct orthonormal transformation.
        
        """ 

        cdef Py_ssize_t i,j,k

        cdef double x = 1+8*params.shape[0]
        cdef int n = <int> cfloor((1+csqrt(x))/2)
        cdef int info
        cdef int* ipiv = <int*> malloc(sizeof(int)*n)
        
        cdef double[:,::1] ima = np.empty((n,n), dtype=np.float64)
        cdef np.ndarray[double, ndim=2, mode='c']  res = np.empty((n,n), dtype=np.float64)

        k = 0
        for i in range(n):
            ima[i,i] = 1.
            for j in range(i+1,n):

                ima[i,j] =  params[k]
                ima[j,i] = -params[k]

                k += 1

        cdef double[:,::1] ipa = ima.T.copy()

        # scipy.linalg.cython_lapack.dgesv(&n,&n,&ima[0,0],&n,ipiv,&ipa[0,0],&n,&info)
        ipa = np.linalg.solve(ima, ipa)
        scipy.linalg.cython_blas.dgemm(transn,transn,&n,&n,&n,&one_double,&ipa[0,0],&n,&ipa[0,0],&n,&zero_double,&res[0,0],&n)

        free(ipiv)

        return res

    @cython.final
    @staticmethod
    def BuildCayleyGraph(Py_ssize_t nbody, Py_ssize_t geodim, list GeneratorList = [], Py_ssize_t max_layers = 1000, bint add_edge_data = False):
        """ Builds the `Cayley graph <https://en.wikipedia.org/wiki/Cayley_graph>`_ of a list of group generators.

        Parameters
        ----------
        nbody : :class:`python:int`
            Number of bodies in the system.
        geodim : :class:`python:int`
            Dimension of ambiant space.
        GeneratorList : :class:`python:list` of :class:`choreo.ActionSym`, optional
            List of generators, by default [].
        max_layers : :class:`python:int`, optional
            Maximum number of layers in the graph before raising an error, by default 1000.
        add_edge_data : :class:`python:bool`, optional
            Whether to add the generator to edges of the graph, by default :data:`python:False`.
        
        Returns
        -------
        :class:`networkx:networkx.DiGraph`
            The Cayley graph.

        Raises
        ------
        ValueError
            If the number of layers in the graph is larger than ``max_layers``.

        """    

        cdef Py_ssize_t i_layer

        alphabet = string.ascii_lowercase

        assert len(GeneratorList) < len(alphabet)

        Graph = networkx.DiGraph()
        Sym = ActionSym.Identity(nbody, geodim)
        Graph.add_node("", Sym=Sym)

        HangingNodesDict = {"":Sym}

        for i_layer in range(max_layers):

            HangingNodesDict = BuildOneCayleyLayer(Graph, GeneratorList, HangingNodesDict, alphabet, add_edge_data)
            
            if len(HangingNodesDict) == 0:
                break
        
        else:

            raise ValueError('Exceeded maximum number of iterations in BuildCayleyGraph')

        return Graph

def BuildOneCayleyLayer(Graph, list GeneratorList, dict HangingNodesDict, alphabet, bint add_edge_data = False):

    cdef ActionSym GenSym, HSym, NewSym, Sym
    cdef Py_ssize_t i, j

    cdef dict NewHangingNodesDict = dict()
    cdef list NextLayer = []
    cdef list UniqueNextLayer = []

    for hkey, HSym in HangingNodesDict.items():

        for i, GenSym in enumerate(GeneratorList):

            NewSym = GenSym.Compose(HSym)

            for key, Sym in Graph.nodes.data("Sym"):
                if NewSym.IsSame(Sym):
                    if add_edge_data:
                        Graph.add_edge(hkey, key, GenSym = GenSym)
                    else:
                        Graph.add_edge(hkey, key)

                    break
            else:
                NextLayer.append((NewSym, hkey, alphabet[i]+hkey, GenSym))

    for layer_item in NextLayer:

        NewSym = layer_item[0]

        for next_layer_item in UniqueNextLayer:

            Sym = next_layer_item[0]

            if NewSym.IsSame(Sym):
                next_layer_item[1].append(layer_item[1])
                next_layer_item[2].append(layer_item[2])
                next_layer_item[3].append(layer_item[3])
                break
        else:

            UniqueNextLayer.append((layer_item[0], [layer_item[1]], [layer_item[2]], [layer_item[3]]))

    for NewSym, hkey_list, key_list, GenSym_list in UniqueNextLayer:

        new_keylen = len(key_list[0])        
        new_key = key_list[0]

        for key in key_list[1:]:
            keylen = len(key)
            if keylen < new_keylen:
                new_keylen = keylen
                new_key = key

        Graph.add_node(new_key, Sym=NewSym)

        NewHangingNodesDict[new_key] = NewSym

        for hkey, GenSym in zip(hkey_list, GenSym_list):

            if add_edge_data:
                Graph.add_edge(hkey, new_key, GenSym=GenSym)
            else:
                Graph.add_edge(hkey, new_key)

    return NewHangingNodesDict
