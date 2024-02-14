import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
from libc.complex cimport cexp

import scipy

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



cdef double minusone_double = -1
cdef double ctwopi = 2* np.pi
cdef double complex cminusitwopi = -1j*ctwopi
cdef double complex citwopi = 1j*ctwopi
cdef int int_zero = 0
cdef int int_one = 1
cdef int int_two = 2

@cython.cdivision(True)
cdef void inplace_twiddle(
    double complex[:,:,::1] ifft_b  ,
    long[::1] nnz_k                 ,
    long nint                       ,
    int n_inter                     ,
) noexcept nogil:

    cdef int nppl = ifft_b.shape[2]
    cdef int ncoeff_min_loop_nnz = nnz_k.shape[0]

    cdef double complex w, wo, winter
    cdef double complex w_pow[16] # minimum size of int on all machines.

    cdef int ibit
    cdef int nbit = 1
    cdef long twopow = 1
    cdef bint *nnz_bin 

    cdef Py_ssize_t m, j, i, k

    if ncoeff_min_loop_nnz > 0:

        if nnz_k[ncoeff_min_loop_nnz-1] > 0:

            wo =  cexp(cminusitwopi / nint)
            winter = 1.

            while (twopow < nnz_k[ncoeff_min_loop_nnz-1]) :
                twopow *= 2
                nbit += 1

            nnz_bin  = <bint*> malloc(sizeof(bint)*nbit*ncoeff_min_loop_nnz)
            for j in range(ncoeff_min_loop_nnz):
                for ibit in range(nbit):
                    nnz_bin[ibit + j*nbit] = ((nnz_k[j] >> (ibit)) & 1) # tests if the ibit-th bit of nnz_k[j] is one 

            for m in range(n_inter):

                w_pow[0] = winter
                for ibit in range(nbit-1):
                    w_pow[ibit+1] = w_pow[ibit] * w_pow[ibit] 

                for j in range(ncoeff_min_loop_nnz):
                    
                    w = 1.
                    for ibit in range(nbit):
                        if nnz_bin[ibit + j*nbit]:
                            w *= w_pow[ibit] 

                    for i in range(nppl):
                        ifft_b[m,j,i] *= w

                winter *= wo

            free(nnz_bin)


@cython.cdivision(True)
cpdef void partial_fft_to_pos_slice_1npr(
    double complex[:,:,::1] ifft_b                  ,
    double complex[:,:,::1] params_basis_reoganized ,
    int ncoeff_min_loop                             ,
    long[::1] nnz_k                                 ,
    double[:,::1] pos_slice                         ,
) noexcept nogil:

    cdef int npr = ifft_b.shape[0] - 1
    cdef int ncoeff_min_loop_nnz = nnz_k.shape[0]
    cdef int geodim = params_basis_reoganized.shape[0]
    cdef int nppl = ifft_b.shape[2]    
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_reoganized_r = <double*> &params_basis_reoganized[0,0,0]
    cdef double* ifft_b_r = NULL
    if ncoeff_min_loop_nnz > 0:
        ifft_b_r = <double*> &ifft_b[0,0,0]

    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom

    inplace_twiddle(ifft_b, nnz_k, nint, npr)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &npr, &ndcom, &dfac, params_basis_reoganized_r, &ndcom, ifft_b_r, &ndcom, &zero_double, &pos_slice[0,0], &geodim)





@cython.cdivision(True)
cpdef void partial_fft_to_pos_slice_2npr(
    double complex[:,:,::1] ifft_b                  ,
    double complex[:,:,::1] params_basis_reoganized ,
    int ncoeff_min_loop                             ,
    long[::1] nnz_k                                 ,
    double[:,::1] pos_slice                         ,
) noexcept nogil:

    cdef int n_inter = ifft_b.shape[0] # Cannot be long as it will be an argument to dgemm
    cdef int npr = n_inter - 1
    cdef int ncoeff_min_loop_nnz = nnz_k.shape[0]
    cdef int geodim = params_basis_reoganized.shape[0]
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_reoganized_r = <double*> &params_basis_reoganized[0,0,0]
    cdef double* ifft_b_r = NULL
    if ncoeff_min_loop_nnz > 0:
        ifft_b_r = <double*> &ifft_b[0,0,0]
    
    cdef int nppl = ifft_b.shape[2]
    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom
    cdef int nconj

    cdef double complex w
    cdef Py_ssize_t m, j, i

    inplace_twiddle(ifft_b, nnz_k, nint, n_inter)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_reoganized_r, &ndcom, ifft_b_r, &ndcom, &zero_double, &pos_slice[0,0], &geodim)


    n_inter = npr-1
   
    for j in range(ncoeff_min_loop_nnz):
        w = cexp(citwopi*nnz_k[j]/ncoeff_min_loop)
        for i in range(nppl):
            scipy.linalg.cython_blas.zscal(&n_inter,&w,&ifft_b[1,j,i],&nzcom)

    # Inplace conjugaison
    ifft_b_r += 1 + ndcom
    nconj = n_inter*nzcom
    scipy.linalg.cython_blas.dscal(&nconj,&minusone_double,ifft_b_r,&int_two)

    cdef double complex *ztmp = <double complex*> malloc(sizeof(double complex) * nconj)
    cdef double *dtmp = (<double*> ztmp) + n_inter*ndcom

    ifft_b_r -= 1
    for i in range(n_inter):
        dtmp -= ndcom
        scipy.linalg.cython_blas.dcopy(&ndcom,ifft_b_r,&int_one,dtmp,&int_one)
        ifft_b_r += ndcom

    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_reoganized_r, &ndcom, dtmp, &ndcom, &zero_double, &pos_slice[npr+1,0], &geodim)

    free(ztmp)







# Benchmark of alternatives
@cython.cdivision(True)
cpdef void partial_fft_to_pos_slice_2npr_nocopy(
    double complex[:,:,::1] ifft_b                  ,
    double complex[:,:,::1] params_basis_reoganized ,
    int ncoeff_min_loop                             ,
    long[::1] nnz_k                                 ,
    double[:,::1] pos_slice                         ,
) noexcept nogil:

    cdef int n_inter = ifft_b.shape[0] # Cannot be long as it will be an argument to dgemm
    cdef int npr = n_inter - 1
    cdef int ncoeff_min_loop_nnz = nnz_k.shape[0]
    cdef int geodim = params_basis_reoganized.shape[0]
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_reoganized_r = <double*> &params_basis_reoganized[0,0,0]
    cdef double* ifft_b_r = NULL
    if ncoeff_min_loop_nnz > 0:
        ifft_b_r = <double*> &ifft_b[0,0,0]

    cdef int nppl = ifft_b.shape[2]
    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom
    cdef int nconj

    inplace_twiddle(ifft_b, nnz_k, nint, n_inter)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_reoganized_r, &ndcom, ifft_b_r, &ndcom, &zero_double, &pos_slice[0,0], &geodim)

    cdef double complex w
    cdef Py_ssize_t m, j, i

    for j in range(ncoeff_min_loop_nnz):
        w = cexp(citwopi*nnz_k[j]/ncoeff_min_loop)
        for m in range(1,npr):
            for i in range(nppl):
                ifft_b[m,j,i] *= w


    n_inter = npr-1
    # Inplace conjugaison
    ifft_b_r += 1 + ndcom
    nconj = n_inter*nzcom
    scipy.linalg.cython_blas.dscal(&nconj,&minusone_double,ifft_b_r,&int_two)

    cdef double complex *ztmp = <double complex*> malloc(sizeof(double complex) * nconj)
    cdef double *dtmp = (<double*> ztmp) + n_inter*ndcom

    ifft_b_r -= 1
    for i in range(n_inter):
        dtmp -= ndcom
        scipy.linalg.cython_blas.dcopy(&ndcom,ifft_b_r,&int_one,dtmp,&int_one)
        ifft_b_r += ndcom

    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_reoganized_r, &ndcom, dtmp, &ndcom, &zero_double, &pos_slice[npr+1,0], &geodim)

    free(ztmp)



    
cpdef void DoAllIFFTs(
    double[::1] params_buf          , long[:,::1] params_shapes     , long[::1] params_shifts   ,
    long[::1] nnz_k_buf             , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    double complex[::1] ifft_buf    , long[:,::1] ifft_shapes       , long[::1] ifft_shifts     ,
):

    cdef double [:,:,::1] params
    cdef long[::1] nnz_k
    cdef double complex[:,:,::1] ifft

    cdef int nloop = params_shapes.shape[0]
    cdef int n
    cdef double complex * dest
    cdef Py_ssize_t il, i

    for il in range(nloop):

        if params_shapes[il,1] > 0:

            params = <double[:params_shapes[il,0],:params_shapes[il,1],:params_shapes[il,2]:1]> &params_buf[params_shifts[il]]
            nnz_k = <long[:nnz_k_shapes[il,0]:1]> &nnz_k_buf[nnz_k_shifts[il]]

            if nnz_k.shape[0] > 0:
                if nnz_k[0] == 0:
                    for i in range(params.shape[2]):
                        params[0,0,i] *= 0.5

            ifft = scipy.fft.rfft(params, axis=0, n=2*params.shape[0])

            dest = &ifft_buf[0] + ifft_shifts[il]
            n = ifft_shifts[il+1] - ifft_shifts[il]
            scipy.linalg.cython_blas.zcopy(&n,&ifft[0,0,0],&int_one,dest,&int_one)

            



cpdef void ifft_to_pos_slice(
    double complex[::1] ifft_buf            , long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    double[::1] pos_slice_buf               , long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop, long nnpr,
):

    cdef double complex[:,:,::1] ifft
    cdef double complex[:,:,::1] params_basis
    cdef long[::1] nnz_k
    cdef double[:,::1] pos_slice

    cdef int nloop = ncoeff_min_loop.shape[0]
    cdef Py_ssize_t il, i

    for il in range(nloop):

        if params_basis_shapes[il,1] > 0:

            ifft = <double complex[:ifft_shapes[il,0],:ifft_shapes[il,1],:ifft_shapes[il,2]:1]> &ifft_buf[ifft_shifts[il]]
            params_basis = <double complex[:params_basis_shapes[il,0],:params_basis_shapes[il,1],:params_basis_shapes[il,2]:1]> &params_basis_buf[params_basis_shifts[il]]
            nnz_k = <long[:nnz_k_shapes[il,0]:1]> &nnz_k_buf[nnz_k_shifts[il]]
            pos_slice = <double[:pos_slice_shapes[il,0],:pos_slice_shapes[il,1]:1]> &pos_slice_buf[pos_slice_shifts[il]]

            if nnpr == 1:
                partial_fft_to_pos_slice_1npr(ifft, params_basis, ncoeff_min_loop[il], nnz_k, pos_slice)
            else:
                partial_fft_to_pos_slice_2npr(ifft, params_basis, ncoeff_min_loop[il], nnz_k, pos_slice)



cpdef void params_to_pos_slice(
    double[::1] params_buf                  , long[:,::1] params_shapes         , long[::1] params_shifts   ,
    double complex[::1] ifft_buf            , long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    double[::1] pos_slice_buf               , long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop, long nnpr,
):

    DoAllIFFTs(
        params_buf  , params_shapes , params_shifts ,
        nnz_k_buf   , nnz_k_shapes  , nnz_k_shifts  ,
        ifft_buf    , ifft_shapes   , ifft_shifts   ,
    )

    ifft_to_pos_slice(
        ifft_buf            , ifft_shapes           , ifft_shifts           ,
        params_basis_buf    , params_basis_shapes   , params_basis_shifts   ,
        nnz_k_buf           , nnz_k_shapes          , nnz_k_shifts          ,
        pos_slice_buf       , pos_slice_shapes      , pos_slice_shifts      ,
        ncoeff_min_loop     , nnpr                  ,
    )
