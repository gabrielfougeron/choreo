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
        self                        ,
        Py_ssize_t[::1] BodyPerm    ,
        double[:,::1] SpaceRot      ,
        Py_ssize_t TimeRev          ,
        Py_ssize_t TimeShiftNum     ,
        Py_ssize_t TimeShiftDen     ,
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
    def FromDict(Sym_dict):

        TimeRev = Sym_dict["TimeRev"]

        if isinstance(TimeRev, str):
            if (TimeRev == "True"):
                TimeRev = -1
            elif (TimeRev == "False"):
                TimeRev = 1
            else:
                raise ValueError("TimeRev must be True or False")

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
        Random Returns a random transformation.
        Warning: the underlying density is **NOT** uniform.
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
        If self transforms positions, then self.TimeDerivative() transforms speeds.
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
        return ( 
            self.IsSamePerm(other) and
            self.IsSameRot(other, atol = atol) and
            self.IsSameTimeRev(other) and
            self.IsSameTimeShift(other)
        )
    
    @cython.final
    cpdef bint IsSamePerm(ActionSym self, ActionSym other):

        cdef bint isid = True
        cdef Py_ssize_t ib
        cdef Py_ssize_t nbody = self._BodyPerm.shape[0]

        for ib in range(nbody):
            isid = isid and (self._BodyPerm[ib] == other._BodyPerm[ib])

        return isid  

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
        return self.TimeRev == other.TimeRev
    
    @cython.final
    cpdef bint IsSameTimeShift(ActionSym self, ActionSym other, double atol = default_atol):
        return self.TimeShiftNum * other.TimeShiftDen == self.TimeShiftDen * other.TimeShiftNum    

    @cython.final
    cpdef bint IsSameRotAndTimeRev(ActionSym self, ActionSym other, double atol = default_atol):
        return self.IsSameTimeRev(other) and self.IsSameRot(other, atol = atol)    
    
    @cython.final
    cpdef bint IsSameRotAndTime(ActionSym self, ActionSym other, double atol = default_atol):
        return self.IsSameTimeShift(other) and self.IsSameTimeRev(other) and self.IsSameRot(other, atol = atol)    

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
            
    @cython.final
    def to_dict(ActionSym self):
    # Useful to write to a json file

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
        for p in itertools.permutations(range(n)):
            for i in range(n):
                if p[p[i]] != i:
                    break
            else:
                yield np.array(p, dtype=np.intp)

    @cython.final
    @staticmethod
    def SurjectiveDirectSpaceRot(double[::1] params):
        # Uses the square of Cayley transform for surjectivity
        # T = ((I-A)^-1 (I+A))^2
        # Where A = SkeySym(params)

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

def BuildCayleyGraph(Py_ssize_t nbody, Py_ssize_t geodim, list GeneratorList = [], Py_ssize_t max_layers = 1000):

    cdef Py_ssize_t i_layer

    alphabet = string.ascii_lowercase

    assert len(GeneratorList) < len(alphabet)

    # Graph = networkx.Graph()
    Graph = networkx.DiGraph()
    Sym = ActionSym.Identity(nbody, geodim)
    Graph.add_node("", Sym=Sym)
    HangingNodesDict = {"":Sym}

    for i_layer in range(max_layers):

        HangingNodesDict = BuildOneCayleyLayer(Graph, GeneratorList, HangingNodesDict, alphabet)
        
        if len(HangingNodesDict) == 0:
            break
    
    else:

        raise ValueError('Exceeded maximum number of iterations in BuildCayleyGraph')

    return Graph

def BuildOneCayleyLayer(Graph, list GeneratorList, dict HangingNodesDict, alphabet):

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
                    Graph.add_edge(hkey, key, GenSym = GenSym)
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

            Graph.add_edge(hkey, new_key, GenSym=GenSym)

    return NewHangingNodesDict




        
