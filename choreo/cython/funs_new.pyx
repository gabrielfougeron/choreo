import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
cimport scipy.linalg.cython_blas

import choreo.scipy_plus.linalg
cimport blis.cy


@cython.cdivision(True)
cdef inline long gcd (long a, long b) noexcept nogil:

    cdef long c
    while ( a != 0 ):
        c = a
        a = b % a
        b = c

    return b

cdef double default_atol = 1e-10

@cython.final
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

    @cython.final
    @cython.cdivision(True)
    def __init__(
        self                    ,
        long[::1] BodyPerm      ,
        double[:,::1] SpaceRot  ,
        long TimeRev            ,
        long TimeShiftNum       ,
        long TimeShiftDen       ,
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

    @cython.final
    def __str__(self):

        out  = ""
        out += f"BodyPerm:\n"
        out += f"{np.asarray(self.BodyPerm)}\n"
        out += f"SpaceRot:\n"
        out += f"{np.asarray(self.SpaceRot)}\n"
        out += f"TimeRev: {np.asarray(self.TimeRev)}\n"
        out += f"TimeShift: {self.TimeShiftNum} / {self.TimeShiftDen}"

        return out
    
    @cython.final
    @staticmethod
    def Identity(long nbody, long geodim):
        """
        Identity: Returns the identity transformation
        """        

        return ActionSym(
            BodyPerm  = np.array(range(nbody), dtype = np.int_),
            SpaceRot  = np.identity(geodim, dtype = np.float64),
            TimeRev   = 1,
            TimeShiftNum = 0,
            TimeShiftDen = 1
        )

    @cython.final
    @staticmethod
    def Random(long nbody, long geodim, long maxden = -1):
        """
        Random Returns a random transformation
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

    @cython.final
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

    @cython.final
    cpdef ActionSym TimeDerivative(ActionSym self):
        r"""
        Returns the time derivative of a symmetry transformation.
        If self transforms positions, then self.TimeDerivative() transforms speeds
        """

        cdef double[:, ::1] SpaceRot = self.SpaceRot.copy()
        for i in range(SpaceRot.shape[0]):
            for j in range(SpaceRot.shape[1]):
                SpaceRot[i, j] *= self.TimeRev

        return ActionSym(
            BodyPerm = self.BodyPerm.copy() ,
            SpaceRot = SpaceRot             ,
            TimeRev = self.TimeRev          ,         
            TimeShiftNum = self.TimeShiftNum,
            TimeShiftDen = self.TimeShiftDen,
        )

    @cython.final
    @cython.cdivision(True)
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

        cdef long g = gcd(num,den)
        num = num // g
        den = den // g

        return ActionSym(
            BodyPerm = ComposeBodyPerm,
            SpaceRot = np.matmul(B.SpaceRot,A.SpaceRot),
            TimeRev = trev,
            TimeShiftNum = num,
            TimeShiftDen = den
        )

    @cython.final
    cpdef ActionSym Conjugate(ActionSym A, ActionSym B):
        r"""
        Returns the conjugation of a transformation wrt another transformation.

        A.Conjugate(B) returns the conjugation B o A o B^-1.
        """

        return B.Compose(A.Compose(B.Inverse()))

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
        cdef long ib
        cdef long nbody = self.BodyPerm.shape[0]

        for ib in range(nbody):
            isid = isid and (self.BodyPerm[ib] == ib)

        return isid
    
    @cython.final
    cpdef bint IsIdentityRot(ActionSym self, double atol = default_atol):

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
    cpdef bint IsSame(ActionSym self, other, double atol = default_atol):
        r"""
        Returns True if the two transformations are almost identical.
        """   
        return ((self.Inverse()).Compose(other)).IsIdentity(atol = atol)
    
    @cython.final
    cpdef bint IsSamePerm(ActionSym self, other):
        return ((self.Inverse()).Compose(other)).IsIdentityPerm()    
    
    @cython.final
    cpdef bint IsSameRot(ActionSym self, other, double atol = default_atol):
        return ((self.Inverse()).Compose(other)).IsIdentityRot(atol = atol)    
    
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
    cpdef (long, long) ApplyT(ActionSym self, long tnum, long tden):

        cdef long num = self.TimeRev * (tnum * self.TimeShiftDen - self.TimeShiftNum * tden)
        cdef long den = tden * self.TimeShiftDen
        num = ((num % den) + den) % den

        cdef long g = gcd(num,den)
        num = num // g
        den = den // g

        return  num, den

    @cython.final
    @cython.cdivision(True)
    cpdef (long, long) ApplyTSegm(ActionSym self, long tnum, long tden):

        if (self.TimeRev == -1): 
            tnum = ((tnum+1)%tden)

        return self.ApplyT(tnum, tden)



cdef double one_double = 1.
cdef double zero_double = 0.
cdef char *transn = 'n'
cdef char *transt = 't'

cdef inline void _blas_matmul_contiguous(
    double[:,::1] a,
    double[:,::1] b,
    double[:,::1] c
) noexcept nogil:

    # nk,km -> nm
    cdef int n = a.shape[0]
    cdef int k = a.shape[1]
    cdef int m = b.shape[1]

    scipy.linalg.cython_blas.dgemm(transn, transn, &m, &n, &k, &one_double, &b[0,0], &m, &a[0,0], &k, &zero_double, &c[0,0], &m)

cpdef void blas_matmul_contiguous(
    double[:,::1] a,
    double[:,::1] b,
    double[:,::1] c
) noexcept nogil:

    cdef int n = a.shape[0]
    cdef int k = a.shape[1]
    cdef int m = b.shape[1]

    scipy.linalg.cython_blas.dgemm(transn, transn, &m, &n, &k, &one_double, &b[0,0], &m, &a[0,0], &k, &zero_double, &c[0,0], &m)


cpdef void blas_matmulTT_contiguous(
    double[:,::1] a,
    double[:,::1] b,
    double[:,::1] c
) noexcept nogil:

    cdef int n = a.shape[1]
    cdef int k = a.shape[0]
    cdef int m = b.shape[0]

    scipy.linalg.cython_blas.dgemm(transt, transt, &m, &n, &k, &one_double, &b[0,0], &k, &a[0,0], &n, &zero_double, &c[0,0], &m)



# The function partial_fft_to_pos_slice implements the following, but in low level
# pos_slice is assumed already allocated, but can be uninitialized (use np.empty)
# 
# 
# pos_slice = np.einsum('ijk,jkl->li', params_basis_reoganized.real, ifft_b.real) + np.einsum('ijk,jkl->li', params_basis_reoganized.imag, ifft_b.imag)  
# 
# Or, similarly:
# 
# ifft_c = ifft_b.view(dtype=np.float64).reshape(ncom, n_inter,2)
# params_basis_reoganized_c = params_basis_reoganized.view(dtype=np.float64).reshape(geodim, ncom,2)
# 
# pos_slice =  np.matmul(ifft_c[:,:,0].T, params_basis_reoganized_c[:,:,0].T)
# pos_slice += np.matmul(ifft_c[:,:,1].T, params_basis_reoganized_c[:,:,1].T)

cpdef void partial_fft_to_pos_slice_blas(
    double complex[:,:,::1] ifft_b                  ,
    double[:,:,::1] params_basis_reoganized_real    ,
    double[:,:,::1] params_basis_reoganized_imag    ,
    double[:,::1] pos_slice                         ,
) noexcept nogil:

    cdef int ninter = pos_slice.shape[0]
    cdef int geodim = pos_slice.shape[1]

    cdef int ncom =  ifft_b.shape[0] * ifft_b.shape[1]

    cdef double* ifft_b_real = <double*> &ifft_b[0,0,0]
    cdef double* res = &pos_slice[0,0]

    cdef int lda = 2*ncom
    # cdef int lda = 2*ninter
    cdef int ldb = 2*ncom

    scipy.linalg.cython_blas.dgemm(
        transt, transt,
        &ncom, &geodim, &geodim, &one_double,
        &params_basis_reoganized_real[0,0,0], &ncom,
        ifft_b_real, &ncom,
        &zero_double, res, &geodim
    )
    
#     # Pointer addition
#     ifft_b_real += 1
#     params_basis_reoganized_real += 1
# 
#     scipy.linalg.cython_blas.dgemm(
#         transt, transt,
#         &ninter, &geodim, &ncom, &one_double,
#         params_basis_reoganized_real, &lda,
#         ifft_b_real, &ldb,
#         &one_double, res, &geodim
#     )


cpdef void partial_fft_to_pos_slice_blis(
    double complex[:,:,::1] ifft_b                    ,
    double complex[:,:,::1] params_basis_reoganized   ,
    double[:,::1] pos_slice                           ,
) noexcept nogil:

    cdef int ninter = pos_slice.shape[0]
    cdef int geodim = pos_slice.shape[1]

    cdef int ncom =  ifft_b.shape[0] * ifft_b.shape[1]

    cdef double* ifft_b_real = <double*> &ifft_b[0,0,0]
    cdef double* params_basis_reoganized_real = <double*> &params_basis_reoganized[0,0,0]
    cdef double* res = &pos_slice[0,0]

    cdef int lda = 2*ninter
    cdef int ldb = 2*ncom

    blis.cy.gemm(
        blis.cy.TRANSPOSE, blis.cy.TRANSPOSE,
        ninter, geodim, ncom,
        1.0, ifft_b_real, lda, 2,
        params_basis_reoganized_real, ldb, 2 ,
        0.0, res, geodim, 1
    )

    # Pointer addition
    ifft_b_real += 1
    params_basis_reoganized_real += 1

    blis.cy.gemm(
        blis.cy.TRANSPOSE, blis.cy.TRANSPOSE,
        ninter, geodim, ncom,
        1.0, ifft_b_real, lda, 2,
        params_basis_reoganized_real, ldb, 2 ,
        1.0, res, geodim, 1
    )
