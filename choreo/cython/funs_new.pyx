import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
from libc.complex cimport cexp

cimport scipy.linalg.cython_blas
from libc.stdlib cimport malloc, free

import choreo.scipy_plus.linalg


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
        cdef long num = B.TimeRev * A.TimeShiftNum * B.TimeShiftDen + B.TimeShiftNum * A.TimeShiftDen
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
    cpdef (long, long) ApplyTInv(ActionSym self, long tnum, long tden):

        cdef long num = self.TimeRev * (tnum * self.TimeShiftDen - self.TimeShiftNum * tden)
        cdef long den = tden * self.TimeShiftDen
        num = ((num % den) + den) % den

        cdef long g = gcd(num,den)
        num = num // g
        den = den // g

        return  num, den

    @cython.final
    @cython.cdivision(True)
    cpdef (long, long) ApplyTInvSegm(ActionSym self, long tnum, long tden):

        if (self.TimeRev == -1): 
            tnum = ((tnum+1)%tden)

        return self.ApplyTInv(tnum, tden)


    @cython.final
    @cython.cdivision(True)
    cpdef (long, long) ApplyT(ActionSym self, long tnum, long tden):

        cdef long num = self.TimeRev * tnum * self.TimeShiftDen + self.TimeShiftNum * tden
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


    # THIS IS NOT A PERFORMANCE-ORIENTED METHOD
    @cython.final
    def TransformSegment(ActionSym self, in_segm, out):

        np.matmul(in_segm, self.SpaceRot.T,out=out)
        if self.TimeRev == -1:
            out[:,:] = out[::-1,:]


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


cpdef void blas_matmulNT_contiguous(
    double[:,::1] a,
    double[:,::1] b,
    double[:,::1] c
) noexcept nogil:

    cdef int n = a.shape[0]
    cdef int k = a.shape[1]
    cdef int m = b.shape[0]

    scipy.linalg.cython_blas.dgemm(transt, transn, &m, &n, &k, &one_double, &b[0,0], &k, &a[0,0], &k, &zero_double, &c[0,0], &m)



cdef double ctwopi = 2* np.pi
cdef double complex cminustwopi = -1j*ctwopi
cdef int int_one = 1

@cython.cdivision(True)
cpdef void partial_fft_to_pos_slice(
    double complex[:,:,::1] ifft_b                  ,
    double complex[:,:,::1] params_basis_reoganized ,
    int ncoeff_min_loop                             ,
    long[::1] nnz_k                                 ,
    double[:,::1] param_basis_0                     ,
    double[:,:,::1] params_loop                     ,
    double[:,::1] pos_slice                         ,
) noexcept nogil:

    cdef int n_inter = ifft_b.shape[0]
    cdef int npr = n_inter -1
    cdef int nppl = ifft_b.shape[2]
    cdef int ncoeff_min_loop_nnz = nnz_k.shape[0]
    cdef int geodim = params_basis_reoganized.shape[0]
    cdef int nint = 2*ncoeff_min_loop*npr
    cdef Py_ssize_t m, j, i, k

    cdef double complex fac
    cdef double complex w, wo, winter

    cdef double* ifft_b_r = <double*> &ifft_b[0,0,0]
    cdef double* params_basis_reoganized_r = <double*> &params_basis_reoganized[0,0,0]
    cdef int ncom = 2*ncoeff_min_loop_nnz*nppl
    cdef double* meanval

    # Compute twiddle factors    
    if ncoeff_min_loop_nnz > 0:

        fac = 1./(npr * ncoeff_min_loop)
        wo =  cexp(cminustwopi / nint)
        winter = 1.

        for m in range(n_inter):

            w = fac

            for i in range(nnz_k[0]):
                w *= winter

            for j in range(ncoeff_min_loop_nnz-1):

                # w = fac * cexp((-1j*ctwopi*nnz_k[j] * m)/nint)

                for i in range(nppl):
                    ifft_b[m,j,i] *= w

                for i in range(nnz_k[j], nnz_k[j+1]):
                    w *= winter

            j = ncoeff_min_loop_nnz-1

            for i in range(nppl):
                ifft_b[m,j,i] *= w

            winter *= wo            

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ncom, &one_double, params_basis_reoganized_r, &ncom, ifft_b_r, &ncom, &zero_double, &pos_slice[0,0], &geodim)

    # Taking care of the mean value
    if ncoeff_min_loop_nnz > 0:
        if nnz_k[0] == 0:

            meanval  = <double*> malloc(sizeof(double)*geodim)

            scipy.linalg.cython_blas.dgemv(transt,&nppl,&geodim,&one_double,&param_basis_0[0,0],&nppl,&params_loop[0,0,0],&int_one,&zero_double,meanval,&int_one)

            for i in range(geodim):
                meanval[i] = - meanval[i] / nint

            for j in range(n_inter):
                for i in range(geodim):
                    pos_slice[j,i] += meanval[i]

            free(meanval)

# @cython.cdivision(True)
# cpdef void Populate_allsegmpos_cy(
#     double[:,:,::1] all_pos         ,
#     double[:,:,::1] allsegmpos      ,
#     double[:,:,::1] GenSpaceRot     ,
#     long[::1]       GenTimeRev      ,
#     long[::1]       gensegm_to_body ,
#     long[::1]       gensegm_to_iint ,
#     long[::1]       BodyLoop        ,
# ) noexcept nogil:
#     
#     cdef int nsegm = allsegmpos.shape[0]
#     cdef int segm_size = allsegmpos.shape[1]
#     cdef int nint = all_pos.shape[1]
# 
#     cdef Py_ssize_t isegm, ib, iint, ibeg, iend
# 
#     bint tmp_alloc = False
# 
#     for isegm in range(nsegm):
# 
#         ib = gensegm_to_body[isegm]
#         iint = gensegm_to_iint[isegm]
#         il = BodyLoop[ib]
#     
#         ibeg = iint * segm_size         
#         iend = ibeg + segm_size
# 
#         if GenTimeRev[isegm] > 0:
# 
#             np.matmul(
#                 all_pos[il,ibeg:iend,:]     ,
#                 GenSpaceRot[isegm,:,:].T    ,
#                 out=allsegmpos[isegm,:,:]   ,
#             )
# 
# 
#     double[:,::1] a,
#     double[:,::1] b,
#     double[:,::1] c
# ) noexcept nogil:
# 
#     cdef int n = a.shape[0]
#     cdef int k = a.shape[1]
#     cdef int m = b.shape[0]
# 
#     scipy.linalg.cython_blas.dgemm(transt, transn, &m, &n, &k, &one_double, &b[0,0], &k, &a[0,0], &k, &zero_double, &c[0,0], &m)
# 


        
#         else:
# 
#         if GenTimeRev[isegm] < 0:
#             allsegmpos[isegm,:,:] = allsegmpos[isegm,::-1,:]