import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
from libc.complex cimport cexp

cimport scipy.linalg.cython_blas
from libc.stdlib cimport malloc, free

from choreo.scipy_plus.cython.blas_consts cimport *

import choreo.scipy_plus.linalg
from choreo.NBodySyst_build import *

import scipy

@cython.final
cdef class NBodySyst():
    r"""
    This class defines a N-body system
    """
    
    cdef readonly long geodim
    cdef readonly long nbody
    cdef readonly long nint_min
    cdef readonly long nloop
    cdef readonly long nsegm

    cdef long[::1] _loopnb
    @property
    def loopnb(self):
        return np.asarray(self._loopnb)

    cdef long[::1] _bodyloop
    @property
    def bodyloop(self):
        return np.asarray(self._bodyloop)

    cdef double[::1] _loopmass
    @property
    def loopmass(self):
        return np.asarray(self._loopmass)

    cdef long[:,::1] _Targets
    @property
    def Targets(self):
        return np.asarray(self._Targets)

    cdef long[:,::1] _bodysegm
    @property
    def bodysegm(self):
        return np.asarray(self._bodysegm)

    cdef long[::1] _loopgen
    @property
    def loopgen(self):
        return np.asarray(self._loopgen)



    cdef readonly object BodyGraph
    cdef readonly object SegmGraph
    cdef readonly object SegmConstraints





    def __init__(
        self                ,
        long geodim         ,
        long nbody          ,
        double[::1] bodymass,
        list Sym_list       , 
    ):

        if (bodymass.shape[0] != nbody):
            raise ValueError(f'Incompatible number of bodies {nbody} vs number of masses {bodymass.shape[0]}')

        self.geodim = geodim
        self.nbody = nbody

        self.nint_min, self.nloop, self._loopnb, self._loopmass, self._bodyloop, self._Targets, self.BodyGraph = DetectLoops(Sym_list, nbody, bodymass)

        self.SegmGraph, self.nint_min, self.nsegm, self._bodysegm, BodyHasContiguousGeneratingSegments, Sym_list = ExploreGlobalShifts_BuildSegmGraph(self.geodim, self.nbody, self.nloop, self._loopnb, self._Targets, self.nint_min, Sym_list)

        self._loopgen = ChooseLoopGen(self.nloop, self._loopnb, BodyHasContiguousGeneratingSegments, self._Targets)

        self.SegmConstraints = AccumulateSegmentConstraints(self.SegmGraph, self.nbody, self.geodim, self.nsegm, self.bodysegm)













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
 
    cdef int n_inter = npr+1
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double* ifft_r = <double*> const_ifft

    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, ifft_r, &ndcom, &zero_double, pos_slice, &geodim)

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
    cdef double* tmp_loc
    cdef double* tmp

    cdef int geodim = GenSpaceRot.shape[1]
    cdef int segm_size_int = segm_size
    cdef int nitems = segm_size_int*geodim
    cdef Py_ssize_t isegm, ib, il, iint
    cdef Py_ssize_t i, idim

    cdef bint NeedsAllocate = False

    for isegm in range(nsegm):
        NeedsAllocate = (NeedsAllocate or (GenTimeRev[isegm] < 0))

    if NeedsAllocate:
        tmp_loc = <double*> malloc(sizeof(double)*nitems)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]

        if GenTimeRev[isegm] == 1:

            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il] + nitems*iint
            segmpos = segmpos_buf_ptr + nitems*isegm

            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_size_int, &geodim, &one_double, &GenSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, segmpos, &geodim)

        else:

            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il] + nitems*iint + geodim
            tmp = tmp_loc

            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_size_int, &geodim, &one_double, &GenSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, tmp, &geodim)

            segmpos = segmpos_buf_ptr + nitems*(isegm+1) - geodim

            for i in range(segm_size):
                for idim in range(geodim):
                    segmpos[idim] = tmp[idim]
                segmpos -= geodim
                tmp += geodim

    if NeedsAllocate:
        free(tmp_loc)

@cython.cdivision(True)
cpdef np.ndarray[double, ndim=3, mode="c"] params_to_segmpos(
    double[::1] params_buf                  , long[:,::1] params_shapes         , long[::1] params_shifts       ,
                                              long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
                                              long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop, long nnpr,
    double[:,:,::1] GenSpaceRot     ,
    long[::1] GenTimeRev            ,
    long[::1] gensegm_to_body       ,
    long[::1] gensegm_to_iint       ,
    long[::1] BodyLoop              ,
    long segm_size                  ,
):

    cdef double complex *ifft_buf_ptr
    cdef double *pos_slice_buf_ptr

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef int geodim = GenSpaceRot.shape[1]
    cdef int size

    cdef double[:,:,::1] segmpos = np.empty((nsegm, segm_size, geodim), dtype=np.float64)

    ifft_buf_ptr = <double complex *> malloc(sizeof(double complex)*ifft_shifts[ifft_shapes.shape[0]])

    params_to_ifft(
        params_buf  , params_shapes , params_shifts ,
        nnz_k_buf   , nnz_k_shapes  , nnz_k_shifts  ,
        ifft_buf_ptr, ifft_shapes   , ifft_shifts   ,
    )

    with nogil:

        size = pos_slice_shifts[pos_slice_shapes.shape[0]]
        pos_slice_buf_ptr = <double *> malloc(sizeof(double)*size)
        scipy.linalg.cython_blas.dscal(&size,&zero_double,pos_slice_buf_ptr,&int_one)

        ifft_to_pos_slice(
            ifft_buf_ptr        , ifft_shapes           , ifft_shifts           ,
            &params_basis_buf[0], params_basis_shapes   , params_basis_shifts   ,
            &nnz_k_buf[0]       , nnz_k_shapes          , nnz_k_shifts          ,
            pos_slice_buf_ptr   , pos_slice_shapes      , pos_slice_shifts      ,
            ncoeff_min_loop     , nnpr                  ,
        )

        free(ifft_buf_ptr)

        pos_slice_to_segmpos(
            pos_slice_buf_ptr   , pos_slice_shapes  , pos_slice_shifts ,
            &segmpos[0,0,0] ,
            GenSpaceRot     ,
            GenTimeRev      ,
            gensegm_to_body ,
            gensegm_to_iint ,
            BodyLoop        ,
            segm_size       ,
        )

        free(pos_slice_buf_ptr)

    return np.asarray(segmpos)
