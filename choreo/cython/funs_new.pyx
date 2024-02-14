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
    const double complex* const_ifft    ,
    long* nnz_k             ,
    long nint               ,
    int n_inter             ,
    int ncoeff_min_loop_nnz ,
    int nppl                ,
) noexcept nogil:

    cdef double complex w, wo, winter
    cdef double complex w_pow[16] # minimum size of int on all machines.

    cdef int ibit
    cdef int nbit = 1
    cdef long twopow = 1
    cdef bint *nnz_bin 

    cdef double complex* ifft = <double complex*> const_ifft

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
                        ifft[0] *= w
                        ifft += 1

                winter *= wo

            free(nnz_bin)


@cython.cdivision(True)
cdef void partial_fft_to_pos_slice_1npr(
    const double complex* const_ifft        ,
    double complex* params_basis            ,  
    long* nnz_k                             ,
    double* pos_slice                       ,
    int npr                                 ,
    int ncoeff_min_loop_nnz                 ,
    int ncoeff_min_loop                     ,
    int geodim                              ,
    int nppl                                ,
) noexcept nogil:
 
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double* ifft_r = <double*> const_ifft

    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom

    inplace_twiddle(const_ifft, nnz_k, nint, npr, ncoeff_min_loop_nnz, nppl)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &npr, &ndcom, &dfac, params_basis_r, &ndcom, ifft_r, &ndcom, &zero_double, pos_slice, &geodim)





@cython.cdivision(True)
cdef void partial_fft_to_pos_slice_2npr(
    const double complex* const_ifft        ,
    double complex* params_basis            ,
    long* nnz_k                             ,
    const double* const_pos_slice           ,
    int npr                                 ,
    int ncoeff_min_loop_nnz                 ,
    int ncoeff_min_loop                     ,
    int geodim                              ,
    int nppl                                ,
) noexcept nogil:

    cdef int n_inter = npr+1
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double complex* ifft = const_ifft
    cdef double* ifft_r = <double*> const_ifft
    cdef double* pos_slice = const_pos_slice
    
    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom
    cdef int nconj

    cdef double complex w
    cdef Py_ssize_t m, j, i

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, ifft_r, &ndcom, &zero_double, pos_slice, &geodim)


    n_inter = npr-1
    ifft += nzcom
    for j in range(ncoeff_min_loop_nnz):
        w = cexp(citwopi*nnz_k[j]/ncoeff_min_loop)
        for i in range(nppl):
            # scipy.linalg.cython_blas.zscal(&n_inter,&w,&ifft[1,j,i],&nzcom)
            scipy.linalg.cython_blas.zscal(&n_inter,&w,ifft,&nzcom)
            ifft += 1

    # Inplace conjugaison
    ifft_r += 1 + ndcom
    nconj = n_inter*nzcom
    scipy.linalg.cython_blas.dscal(&nconj,&minusone_double,ifft_r,&int_two)

    cdef double complex *ztmp = <double complex*> malloc(sizeof(double complex) * nconj)
    cdef double *dtmp = (<double*> ztmp) + n_inter*ndcom

    ifft_r -= 1
    for i in range(n_inter):
        dtmp -= ndcom
        scipy.linalg.cython_blas.dcopy(&ndcom,ifft_r,&int_one,dtmp,&int_one)
        ifft_r += ndcom

    pos_slice += (npr+1)*geodim
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, dtmp, &ndcom, &zero_double, pos_slice, &geodim)

    free(ztmp)







    
cdef void params_to_ifft(
    double[::1] params_buf          , long[:,::1] params_shapes     , long[::1] params_shifts   ,
    long[::1] nnz_k_buf             , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    double complex *ifft_buf_ptr    , long[:,::1] ifft_shapes       , long[::1] ifft_shifts     ,
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

            dest = ifft_buf_ptr + ifft_shifts[il]
            n = ifft_shifts[il+1] - ifft_shifts[il]
            scipy.linalg.cython_blas.zcopy(&n,&ifft[0,0,0],&int_one,dest,&int_one)


cdef void ifft_to_pos_slice(
    double complex *ifft_buf_ptr            , long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex *params_basis_buf_ptr    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long* nnz_k_buf_ptr                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    double* pos_slice_buf_ptr               , long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop, long nnpr,
) noexcept nogil:

    cdef double complex* ifft
    cdef double complex* params_basis
    cdef long* nnz_k
    cdef double* pos_slice

    cdef int nloop = ncoeff_min_loop.shape[0]
    cdef Py_ssize_t il, i

    cdef int npr
    cdef int ncoeff_min_loop_il
    cdef int ncoeff_min_loop_nnz
    cdef int geodim
    cdef int nppl

    for il in range(nloop):

        if params_basis_shapes[il,1] > 0:

            ifft = ifft_buf_ptr + ifft_shifts[il]
            params_basis = params_basis_buf_ptr + params_basis_shifts[il]
            nnz_k = nnz_k_buf_ptr + nnz_k_shifts[il]
            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il]

            npr = ifft_shapes[il,0] - 1
            ncoeff_min_loop_nnz = nnz_k_shapes[il,0]
            ncoeff_min_loop_il = ncoeff_min_loop[il]
            geodim = params_basis_shapes[il,0]
            nppl = ifft_shapes[il,2] 

            if nnpr == 1:
                partial_fft_to_pos_slice_1npr(
                    ifft, params_basis, nnz_k, pos_slice,
                    npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                )
            else:
                partial_fft_to_pos_slice_2npr(
                    ifft, params_basis, nnz_k, pos_slice,
                    npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                )

cdef void pos_slice_to_segmpos(
    const double* pos_slice_buf_ptr , long[:,::1] pos_slice_shapes  , long[::1] pos_slice_shifts    ,
    const double* segmpos_buf_ptr   ,
    double[:,:,::1] GenSpaceRot     ,
    long[::1] GenTimeRev            ,
    long[::1] gensegm_to_body       ,
    long[::1] gensegm_to_iint       ,
    long[::1] BodyLoop              ,
    long segm_size                  ,
) noexcept nogil:


    cdef int nsegm = gensegm_to_body.shape[0]
    cdef double* pos_slice
    cdef double* segmpos
    cdef double* tmp

    cdef int geodim = GenSpaceRot.shape[1]
    cdef int nitems = segm_size*geodim
    cdef Py_ssize_t isegm, ib, il
    cdef Py_ssize_t i

    cdef bint NeedsAllocate = False

    for isegm in range(nsegm):
        NeedsAllocate = NeedsAllocate || (GenTimeRev[isegm] < 0)

    if NeedsAllocate:
        tmp = <double*> malloc(sizeof(double)*nitems)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]

        pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il] + nitems*iint
    
        if GenTimeRev[isegm] == 1:

            segmpos = segmpos_buf_ptr + nitems*isegm
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_size, &geodim, &one_double, &GenSpaceRot[isegm,0,0], &k, pos_slice, &geodim, &zero_double, segmpos, &geodim)

        else:

            segmpos = segmpos_buf_ptr + nitems*(isegm+1)
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_size, &geodim, &one_double, &GenSpaceRot[isegm,0,0], &k, pos_slice, &geodim, &zero_double, tmp, &geodim)

            for i in range(nitems):
                segmpos -= 1
                segmpos[0] = tmp[0]

    if NeedsAllocate:
        free(tmp)


cpdef void params_to_pos_slice(
    double[::1] params_buf                  , long[:,::1] params_shapes         , long[::1] params_shifts       ,
                                              long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    double[::1] pos_slice_buf               , long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop, long nnpr,
):

    cdef double complex *ifft_buf_ptr = <double complex *> malloc(sizeof(double complex)*ifft_shifts[ifft_shapes.shape[0]])

    params_to_ifft(
        params_buf  , params_shapes , params_shifts ,
        nnz_k_buf   , nnz_k_shapes  , nnz_k_shifts  ,
        ifft_buf_ptr, ifft_shapes   , ifft_shifts   ,
    )

    with nogil:
        ifft_to_pos_slice(
            ifft_buf_ptr        , ifft_shapes           , ifft_shifts           ,
            &params_basis_buf[0], params_basis_shapes   , params_basis_shifts   ,
            &nnz_k_buf[0]       , nnz_k_shapes          , nnz_k_shifts          ,
            &pos_slice_buf[0]   , pos_slice_shapes      , pos_slice_shifts      ,
            ncoeff_min_loop     , nnpr                  ,
        )

    free(ifft_buf_ptr)

@cython.cdivision(True)
cpdef void params_to_segmpos(
    double[::1] params_buf                  , long[:,::1] params_shapes         , long[::1] params_shifts       ,
                                              long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
                                              long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop, long nnpr,
):

    cdef double complex *ifft_buf_ptr
    cdef double *pos_slice_buf_ptr


    ifft_buf_ptr = <double complex *> malloc(sizeof(double complex)*ifft_shifts[ifft_shapes.shape[0]])

    params_to_ifft(
        params_buf  , params_shapes , params_shifts ,
        nnz_k_buf   , nnz_k_shapes  , nnz_k_shifts  ,
        ifft_buf_ptr, ifft_shapes   , ifft_shifts   ,
    )

    with nogil:

        pos_slice_buf_ptr = <double *> malloc(sizeof(double)*pos_slice_shifts[pos_slice_shapes.shape[0]])

        ifft_to_pos_slice(
            ifft_buf_ptr        , ifft_shapes           , ifft_shifts           ,
            params_basis_buf    , params_basis_shapes   , params_basis_shifts   ,
            nnz_k_buf           , nnz_k_shapes          , nnz_k_shifts          ,
            pos_slice_buf_ptr   , pos_slice_shapes      , pos_slice_shifts      ,
            ncoeff_min_loop     , nnpr                  ,
        )

        free(ifft_buf_ptr)

        # cdef double *segmpos_buf_ptr = <double *> malloc(sizeof(double)*pos_slice_shifts[pos_slice_shapes.shape[0]])


        pos_slice_to_segmpos(
            double[::1] params_buf                  , long[:,::1] params_shapes         , long[::1] params_shifts       ,
                                                    long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
            double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
            long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
                                                    long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
            long[::1] ncoeff_min_loop, long nnpr,
        ):

        free(pos_slice_buf_ptr)

