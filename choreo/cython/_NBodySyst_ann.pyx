from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport log as clog
from libc.math cimport sqrt as csqrt
from libc.math cimport exp as cexp
from libc.math cimport cos as ccos
from libc.math cimport sin as csin

import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.stdlib cimport malloc, free, rand
from libc.string cimport memset
from libc.stdint cimport uintptr_t

cdef extern from "limits.h":
    int RAND_MAX
cdef extern from "float.h":
    double DBL_MAX

cimport scipy.linalg.cython_blas
from choreo.scipy_plus.cython.blas_consts cimport *

from choreo.cython._ActionSym cimport ActionSym

import scipy

try:
    import mkl_fft
    MKL_FFT_AVAILABLE = True
except:
    MKL_FFT_AVAILABLE = False

try:
    import ducc0
    DUCC_FFT_AVAILABLE = True
except:
    DUCC_FFT_AVAILABLE = False

from choreo.cython.optional_pyfftw cimport pyfftw

cdef int USE_SCIPY_FFT = 0
cdef int USE_MKL_FFT = 1
cdef int USE_FFTW_FFT = 2
cdef int USE_DUCC_FFT = 3

cdef int GENERAL_SYM = 0
cdef int RFFT = 1

@cython.profile(False)
@cython.linetrace(False)
cdef inline void inline_gravity_pot(double xsq, double* res) noexcept nogil:
    
    cdef double a = cpow(xsq,-2.5)
    cdef double b = xsq*a

    res[0] = -xsq*b
    res[1]= 0.5*b
    res[2] = (-0.75)*a

@cython.profile(False)
@cython.linetrace(False)
cdef void gravity_pot(double xsq, double* res, void* pot_params) noexcept nogil:
    inline_gravity_pot(xsq, res)

cdef void power_law_pot(double xsq, double* res, void* pot_params) noexcept nogil:

    cdef double* pot_params_d = <double*> pot_params
    cdef double a = cpow(xsq, pot_params_d[2])
    cdef double b = xsq*a

    res[0] = -xsq*b
    res[1] = pot_params_d[0]*b
    res[2] = pot_params_d[1]*a

cdef double[::1] default_Hash_exp = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9])

ctypedef struct ODE_params_t:

    Py_ssize_t          nbin
    Py_ssize_t          geodim
    Py_ssize_t          nsegm
    Py_ssize_t*         BinSourceSegm_ptr   
    Py_ssize_t*         BinTargetSegm_ptr 
    double*             InvSegmMass
    double*             SegmCharge
    double*             BinSpaceRot_ptr     
    bint*               BinSpaceRotIsId_ptr     
    double*             BinProdChargeSum_ptr    
    double*             BinProdChargeSumSource_ptr    
    double*             BinProdChargeSumTarget_ptr    
    inter_law_fun_type  inter_law   
    void*               inter_law_param_ptr

@cython.cdivision(True)
cdef void Make_Init_bounds_coeffs(
    double *params_pos_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    double coeff_ampl_o             , double coeff_ampl_min             ,
    Py_ssize_t k_infl               , Py_ssize_t k_max                  ,
) noexcept nogil:

    cdef double* cur_param_pos_buf
    cdef double ampl
    cdef double randlimfac = 0.1

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam
    cdef Py_ssize_t k, ko

    cdef double coeff_slope = clog(cfabs(coeff_ampl_o/coeff_ampl_min))/(k_max-k_infl)

    for il in range(nloop):

        cur_param_pos_buf = params_pos_buf + 2*params_shifts[il]

        for ipr in range(params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                if (k <= k_infl):
                    ampl = coeff_ampl_o
                else:
                    ampl = coeff_ampl_o * cexp(-coeff_slope*(k-k_infl))

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = ampl
                    # cur_param_pos_buf[0] = ampl * (1. + 0.1*(<double> rand()) / (<double> RAND_MAX))
                    cur_param_pos_buf += 1

@cython.cdivision(True)
cdef void changevar_mom_pos(
    double *params_mom_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double **params_pos_buf         , 
) noexcept nogil:

    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double* cur_param_pos_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam, nmem
    cdef Py_ssize_t k, ko

    for il in range(nloop):

        cur_param_pos_buf = params_pos_buf[il]
        nmem = params_shifts[il+1]-params_shifts[il]
        memset(cur_param_pos_buf + nmem, 0, sizeof(double)*nmem)

        loopmul = 1./csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_param_pos_buf[0] = loopmul*cur_params_mom_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1
                    
                    else:
                        # DO NOT INCREMENT cur_param_mom_buf !!!
                        cur_param_pos_buf[0] = 0
                        cur_param_pos_buf += 1

            else:

                mul = loopmul/k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul/k

            for iparam in range(params_shapes[il,2]):

                cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul/k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1
 
@cython.cdivision(True)
cdef void changevar_mom_pos_invT(
    double *params_mom_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double **params_pos_buf         , 
) noexcept nogil:

    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double* cur_param_pos_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam, nmem
    cdef Py_ssize_t k, ko

    for il in range(nloop):

        loopmul = csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        cur_param_pos_buf = params_pos_buf[il]
        nmem = params_shifts[il+1]-params_shifts[il]
        memset(cur_param_pos_buf + nmem, 0, sizeof(double)*nmem)

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_param_pos_buf[0] = loopmul*cur_params_mom_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1
                    
                    else:
                        # DO NOT INCREMENT cur_param_mom_buf !!!
                        cur_param_pos_buf[0] = 0
                        cur_param_pos_buf += 1

            else:

                mul = loopmul*k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul*k

            for iparam in range(params_shapes[il,2]):

                cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul*k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

cdef void changevar_mom_pos_inv(
    double **params_pos_buf         , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double *params_mom_buf          , 
) noexcept nogil:

    cdef double* cur_param_pos_buf
    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam
    cdef Py_ssize_t k, ko

    for il in range(nloop):

        loopmul = csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        cur_param_pos_buf = params_pos_buf[il]

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_params_mom_buf[0] = loopmul*cur_param_pos_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1

                    else:
                        # DO NOT INCREMENT cur_params_mom_buf !!!
                        cur_param_pos_buf += 1

            else:

                mul = loopmul * k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul * k

            for iparam in range(params_shapes[il,2]):

                cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul * k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

@cython.cdivision(True)
cdef void changevar_mom_pos_T(
    double **params_pos_buf         , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double *params_mom_buf          ,   
) noexcept nogil:

    cdef double* cur_param_pos_buf
    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam
    cdef Py_ssize_t k, ko

    for il in range(nloop):

        loopmul = 1./csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        cur_param_pos_buf = params_pos_buf[il]

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_params_mom_buf[0] = loopmul*cur_param_pos_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1

                    else:
                        # DO NOT INCREMENT cur_params_mom_buf !!!
                        cur_param_pos_buf += 1

            else:

                mul = loopmul / k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul / k

            for iparam in range(params_shapes[il,2]):

                cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul / k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

@cython.cdivision(True)
cdef void changevar_mom_vel(
    double *params_mom_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double **params_vel_buf         , 
) noexcept nogil:

    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double* cur_param_vel_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam, nmem
    cdef Py_ssize_t k, ko

    for il in range(nloop):

        cur_param_vel_buf = params_vel_buf[il]
        nmem = params_shifts[il+1]-params_shifts[il]
        memset(cur_param_vel_buf + nmem, 0, sizeof(double)*nmem)

        loopmul = 1./csqrt(loopnb[il] * loopmass[il])

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_param_vel_buf[0] = 0

                        cur_params_mom_buf += 1
                        cur_param_vel_buf += 1
                    
                    else:
                        # DO NOT INCREMENT cur_param_mom_buf !!!
                        cur_param_vel_buf[0] = 0
                        cur_param_vel_buf += 1

            else:

                mul = loopmul

                for iparam in range(params_shapes[il,2]):

                    cur_param_vel_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_vel_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul

            for iparam in range(params_shapes[il,2]):

                cur_param_vel_buf[0] = mul*cur_params_mom_buf[0]

                cur_params_mom_buf += 1
                cur_param_vel_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul

                for iparam in range(params_shapes[il,2]):

                    cur_param_vel_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_vel_buf += 1
 

@cython.cdivision(True)
cdef double params_to_kin_nrg(
    double *params_mom_buf      , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] ncor_loop   , Py_ssize_t[::1] nco_in_loop       ,
) noexcept nogil:

    cdef double* loc = params_mom_buf
    cdef double loopmul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il

    cdef double kin = 0
    cdef int beg
    cdef int n

    for il in range(nloop):

        loc += ncor_loop[il]
        n = params_shifts[il+1] - (params_shifts[il] + ncor_loop[il] + nco_in_loop[il])
        
        kin += scipy.linalg.cython_blas.ddot(&n, loc, &int_one, loc, &int_one)

        loc += n

    kin *= 0.5

    return kin
 
@cython.cdivision(True)
cdef void params_to_kin_nrg_grad_daxpy(
    double *params_mom_buf      , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] ncor_loop   , Py_ssize_t[::1] nco_in_loop       ,
    double mul                  ,
    double *grad_buf            ,
) noexcept nogil:

    cdef double* loc = params_mom_buf
    cdef double* grad_loc = grad_buf

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il

    cdef double kin = 0
    cdef int beg
    cdef int n

    for il in range(nloop):

        loc += ncor_loop[il]
        grad_loc += ncor_loop[il]
        n = params_shifts[il+1] - (params_shifts[il] + ncor_loop[il] + nco_in_loop[il])
        
        scipy.linalg.cython_blas.daxpy(&n, &mul, loc, &int_one, grad_loc, &int_one)

        loc += n
        grad_loc += n

@cython.cdivision(True)
cdef void inplace_twiddle(
    double complex* const_ifft  ,
    Py_ssize_t* nnz_k           ,
    Py_ssize_t nint             ,
    int n_inter                 ,
    int ncoeff_min_loop_nnz     ,
    int nppl                    ,
    int direction               , # -1 or 1
) noexcept nogil:

    cdef double complex w, wo, winter
    cdef double complex w_pow[16] # minimum size of int on all machines.

    cdef double arg

    cdef int ibit
    cdef int nbit = 1
    cdef Py_ssize_t twopow = 1
    cdef bint *nnz_bin 

    cdef double complex* ifft = <double complex*> const_ifft

    cdef Py_ssize_t m, j, i, k

    if ncoeff_min_loop_nnz > 0:

        if nnz_k[ncoeff_min_loop_nnz-1] > 0:

            arg = direction * ctwopi / nint
            wo =  ccos(arg) + 1j*csin(arg)
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

                    # w = ccexp(direction * citwopi * m * nnz_k[j] / nint)

                    for i in range(nppl):
                        ifft[0] *= w
                        ifft += 1

                winter *= wo

            free(nnz_bin)

@cython.cdivision(True)
cdef void partial_fft_to_pos_slice_2_sub(
    double complex* const_ifft      ,
    double complex* params_basis    ,  
    Py_ssize_t* nnz_k               ,
    double* pos_slice               ,
    int npr                         ,
    int ncoeff_min_loop_nnz         ,
    int ncoeff_min_loop             ,
    int geodim                      ,
    int nppl                        ,
) noexcept nogil:
 
    cdef int n_inter = npr+1
    cdef Py_ssize_t nint = 2*ncoeff_min_loop*npr

    cdef double dfac = 2.

    # Casting double complex to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double* ifft_r = <double*> const_ifft

    cdef int ndcom = 2*ncoeff_min_loop_nnz*nppl

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, -1)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, ifft_r, &ndcom, &zero_double, pos_slice, &geodim)

@cython.cdivision(True)
cdef void pos_slice_to_partial_fft_2_sub(
    double* pos_slice               ,
    double complex* params_basis    ,  
    Py_ssize_t* nnz_k               ,
    double complex* const_ifft      ,
    int npr                         ,
    int ncoeff_min_loop_nnz         ,
    int ncoeff_min_loop             ,
    int geodim                      ,
    int nppl                        ,
    int direction                   ,
) noexcept nogil:
 
    cdef int n_inter = npr+1
    cdef Py_ssize_t nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting double complex to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double* ifft_r = <double*> const_ifft

    cdef int ndcom = 2*ncoeff_min_loop_nnz*nppl

    if direction > 0:
        dfac = 1.
    else:
        dfac = 2*npr

    # Again with the clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transn, transn, &ndcom, &n_inter, &geodim, &dfac, params_basis_r, &ndcom, pos_slice, &geodim, &zero_double, ifft_r, &ndcom)

    if direction < 0:
        # Double start and end of ifft_r
        scipy.linalg.cython_blas.dscal(&ndcom,&two_double,ifft_r,&int_one)
        ifft_r += npr*ndcom
        scipy.linalg.cython_blas.dscal(&ndcom,&two_double,ifft_r,&int_one)

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, 1)

@cython.cdivision(True)
cdef void partial_fft_to_pos_slice_1_sub(
    double complex* const_ifft      ,
    double complex* params_basis    ,
    Py_ssize_t* nnz_k               ,
    double* const_pos_slice         ,
    int npr                         ,
    int ncoeff_min_loop_nnz         ,
    int ncoeff_min_loop             ,
    int geodim                      ,
    int nppl                        ,
) noexcept nogil:

    cdef int n_inter = npr+1
    cdef Py_ssize_t nint = 2*ncoeff_min_loop*npr

    cdef double dfac = 2.

    # Casting double complex to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double complex* ifft = const_ifft
    cdef double* ifft_r = <double*> const_ifft
    cdef double* pos_slice = const_pos_slice
    
    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom
    cdef int nconj

    cdef double complex w
    cdef double arg
    cdef Py_ssize_t m, j, i

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, -1)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, ifft_r, &ndcom, &zero_double, pos_slice, &geodim)

    n_inter = npr-1
    ifft += nzcom
    for j in range(ncoeff_min_loop_nnz):

        arg = ctwopi*nnz_k[j]/ncoeff_min_loop
        w = ccos(arg) + 1j*csin(arg)

        for i in range(nppl):
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

@cython.cdivision(True)
cdef void pos_slice_to_partial_fft_1_sub(
    double* const_pos_slice         ,
    double complex* params_basis    ,
    Py_ssize_t* nnz_k               ,
    double complex* const_ifft      ,
    int npr                         ,
    int ncoeff_min_loop_nnz         ,
    int ncoeff_min_loop             ,
    int geodim                      ,
    int nppl                        ,
    int direction                   ,
) noexcept nogil:

    cdef int n_inter = npr+1
    cdef Py_ssize_t nint = 2*ncoeff_min_loop*npr

    cdef double dfac
    cdef double arg

    # Casting double complex to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double complex* ifft = const_ifft
    cdef double* ifft_r = <double*> const_ifft
    cdef double* pos_slice
    
    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom
    cdef int nconj

    cdef double complex w
    cdef Py_ssize_t m, j, i

    if direction > 0:
        dfac = 0.5
    else:
        dfac = 2*npr

    n_inter = npr-1
    nconj = n_inter*nzcom

    cdef double complex *ztmp = <double complex*> malloc(sizeof(double complex) * nconj)
    cdef double *dtmp = (<double*> ztmp)

    pos_slice = const_pos_slice + (npr+1)*geodim
    # Again with the clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transn, transn, &ndcom, &n_inter, &geodim, &dfac, params_basis_r, &ndcom, pos_slice, &geodim, &zero_double, dtmp, &ndcom)

    dtmp += n_inter*ndcom
    ifft_r = (<double*> const_ifft) + ndcom
    for i in range(n_inter):
        dtmp -= ndcom
        scipy.linalg.cython_blas.dcopy(&ndcom,dtmp,&int_one,ifft_r,&int_one)
        ifft_r += ndcom

    free(ztmp)

    # Inplace conjugaison
    ifft_r = (<double*> const_ifft) + 1 + ndcom
    scipy.linalg.cython_blas.dscal(&nconj,&minusone_double,ifft_r,&int_two)    

    ifft += nzcom
    for j in range(ncoeff_min_loop_nnz):

        arg = cminustwopi*nnz_k[j]/ncoeff_min_loop
        w = ccos(arg) + 1j*csin(arg)

        for i in range(nppl):
            scipy.linalg.cython_blas.zscal(&n_inter,&w,ifft,&nzcom)
            ifft += 1

    # Initialize start and end of ifft_r
    ifft_r = (<double*> const_ifft)
    memset(ifft_r, 0, sizeof(double)*ndcom)
    ifft_r = (<double*> const_ifft) + npr*ndcom
    memset(ifft_r, 0, sizeof(double)*ndcom)

    pos_slice = const_pos_slice
    n_inter = npr+1
    ifft_r = (<double*> const_ifft) 
    # Again with the clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transn, transn, &ndcom, &n_inter, &geodim, &dfac, params_basis_r, &ndcom, pos_slice, &geodim, &one_double, ifft_r, &ndcom)

    # Double start and end of ifft_r
    scipy.linalg.cython_blas.dscal(&ndcom,&two_double,ifft_r,&int_one)
    ifft_r += npr*ndcom
    scipy.linalg.cython_blas.dscal(&ndcom,&two_double,ifft_r,&int_one)

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, 1)

cdef void Adjust_after_last_gen(
    double** pos_slice_buf_ptr              , Py_ssize_t[::1] pos_slice_shifts  ,
    Py_ssize_t[:,::1] ifft_shapes           ,
    Py_ssize_t[:,::1] params_basis_shapes   ,
    Py_ssize_t[::1] n_sub_fft               ,
    Py_ssize_t[::1] ALG_Iint                ,
    Py_ssize_t[::1] ALG_TimeRev             , double[:,:,::1] ALG_SpaceRot      ,
    Py_ssize_t segm_size                    ,
)noexcept nogil:

    cdef double* pos_slice
    cdef double* pos_slice_uneven_source

    cdef Py_ssize_t nloop = params_basis_shapes.shape[0]
    cdef Py_ssize_t geodim = params_basis_shapes[0,0]
    cdef Py_ssize_t il

    cdef Py_ssize_t npr

    for il in range(nloop):

        if params_basis_shapes[il,1] > 0:

            if (n_sub_fft[il] == 1):

                npr = ifft_shapes[il,0] - 1
                pos_slice = pos_slice_buf_ptr[il] + 2*npr*geodim

                if ALG_TimeRev[il] == 1:
                    pos_slice_uneven_source = pos_slice_buf_ptr[il] + ALG_Iint[il]*segm_size*geodim
                else:
                    pos_slice_uneven_source = pos_slice_buf_ptr[il] + ((ALG_Iint[il]+1)*segm_size - 1)*geodim

                for idim in range(geodim):
                    for jdim in range(geodim):
                        pos_slice[idim] += ALG_SpaceRot[il,idim,jdim] * pos_slice_uneven_source[jdim]

cdef void Adjust_after_last_gen_T(
    double** pos_slice_buf_ptr              , Py_ssize_t[::1] pos_slice_shifts  ,
    Py_ssize_t[:,::1] ifft_shapes           ,
    Py_ssize_t[:,::1] params_basis_shapes   ,
    Py_ssize_t[::1] n_sub_fft               ,
    Py_ssize_t[::1] ALG_Iint                ,
    Py_ssize_t[::1] ALG_TimeRev             , double[:,:,::1] ALG_SpaceRot      ,
    Py_ssize_t segm_size                    ,
)noexcept nogil:

    cdef double* pos_slice
    cdef double* pos_slice_uneven_source

    cdef Py_ssize_t nloop = params_basis_shapes.shape[0]
    cdef Py_ssize_t geodim = params_basis_shapes[0,0]
    cdef Py_ssize_t il

    cdef Py_ssize_t npr

    for il in range(nloop):

        if params_basis_shapes[il,1] > 0:

            if (n_sub_fft[il] == 1):

                npr = ifft_shapes[il,0] - 1
                pos_slice = pos_slice_buf_ptr[il] + 2*npr*geodim

                if ALG_TimeRev[il] == 1:
                    pos_slice_uneven_source = pos_slice_buf_ptr[il] + ALG_Iint[il]*segm_size*geodim
                else:
                    pos_slice_uneven_source = pos_slice_buf_ptr[il] + ((ALG_Iint[il]+1)*segm_size - 1)*geodim

                for idim in range(geodim):
                    for jdim in range(geodim):

                        pos_slice_uneven_source[jdim] += ALG_SpaceRot[il,idim,jdim] * pos_slice[idim]

cdef void params_to_pos_slice(
    double** params_buf                     , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    Py_ssize_t* nnz_k_buf_ptr               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    double complex **ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    int[::1] ParamBasisShortcut             ,
    int fft_backend                         , pyfftw.fftw_exe** fftw_genrfft_exe    , pyfftw.fftw_exe** fftw_symirfft_exe   ,
    double complex *params_basis_buf_ptr    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    double** pos_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             ,
) noexcept nogil:

    cdef Py_ssize_t idim, k, kmax, ddk

    cdef double [:,:,::1] params_mv
    cdef double complex[:,:,::1] ifft_mv
    cdef double complex [:,::1] params_c_mv
    cdef double[:,::1] rfft_mv

    cdef double dfac

    cdef int nloop = params_shapes.shape[0]
    cdef int geodim = params_basis_shapes[0,0]
    cdef int n
    cdef double* buf
    cdef double complex* dest
    cdef Py_ssize_t il, i

    cdef double complex* ifft
    cdef double complex* params_basis
    cdef Py_ssize_t* nnz_k
    cdef double* pos_slice

    cdef int npr
    cdef int ncoeff_min_loop_il
    cdef int ncoeff_min_loop_nnz
    cdef int nppl

    for il in range(nloop):

        memset(pos_slice_buf_ptr[il], 0, sizeof(double)*(pos_slice_shifts[il+1]-pos_slice_shifts[il]))

        if ParamBasisShortcut[il] == GENERAL_SYM:

            if params_shapes[il,1] > 0:

                buf = params_buf[il]

                if nnz_k_shapes[il,0] > 0:
                    if nnz_k_buf_ptr[nnz_k_shifts[il]] == 0:
                        for i in range(params_shapes[il,2]):
                            buf[i] *= 0.5

            if fft_backend == USE_FFTW_FFT:

                if params_shapes[il,1] > 0:
                    pyfftw.execute_in_nogil(fftw_genrfft_exe[il])

            elif fft_backend == USE_MKL_FFT:

                with gil:

                    if params_shapes[il,1] > 0:

                        params_mv = <double[:2*params_shapes[il,0],:params_shapes[il,1],:params_shapes[il,2]:1]> params_buf[il]

                        ifft_mv = mkl_fft._numpy_fft.rfft(params_mv, axis=0)

                        dest = ifft_buf_ptr[il]
                        n = ifft_shifts[il+1] - ifft_shifts[il]
                        scipy.linalg.cython_blas.zcopy(&n,&ifft_mv[0,0,0],&int_one,dest,&int_one)

            elif fft_backend == USE_DUCC_FFT:

                with gil:

                    if params_shapes[il,1] > 0:

                        ducc0.fft.r2c(
                            np.asarray(<double[:2*params_shapes[il,0],:params_shapes[il,1],:params_shapes[il,2]:1]> (params_buf[il])), 
                            axes=[0]    ,
                            out = np.asarray(<double complex[:ifft_shapes[il,0],:ifft_shapes[il,1],:ifft_shapes[il,2]:1]> (ifft_buf_ptr[il])), 
                        )

            elif fft_backend == USE_SCIPY_FFT:

                with gil:

                    if params_shapes[il,1] > 0:

                        params_mv = <double[:2*params_shapes[il,0],:params_shapes[il,1],:params_shapes[il,2]:1]> params_buf[il]

                        ifft_mv = scipy.fft.rfft(params_mv, axis=0, overwrite_x=True)

                        dest = ifft_buf_ptr[il]
                        n = ifft_shifts[il+1] - ifft_shifts[il]
                        scipy.linalg.cython_blas.zcopy(&n,&ifft_mv[0,0,0],&int_one,dest,&int_one)
                        

            if params_basis_shapes[il,1] > 0:

                ifft = ifft_buf_ptr[il]
                params_basis = params_basis_buf_ptr + params_basis_shifts[il]
                nnz_k = nnz_k_buf_ptr + nnz_k_shifts[il]
                pos_slice = pos_slice_buf_ptr[il]

                npr = ifft_shapes[il,0] - 1
                ncoeff_min_loop_nnz = nnz_k_shapes[il,0]
                ncoeff_min_loop_il = ncoeff_min_loop[il]
                nppl = ifft_shapes[il,2] 

                if n_sub_fft[il] == 2:
                    partial_fft_to_pos_slice_2_sub(
                        ifft, params_basis, nnz_k, pos_slice,
                        npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                    )
                else:
                    partial_fft_to_pos_slice_1_sub(
                        ifft, params_basis, nnz_k, pos_slice,
                        npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                    )

        elif ParamBasisShortcut[il] == RFFT:

            if fft_backend == USE_FFTW_FFT:

                pyfftw.execute_in_nogil(fftw_symirfft_exe[il])

            elif fft_backend == USE_MKL_FFT:

                with gil:

                    params_c_mv = <double complex[:(params_shapes[il,0]+1),:geodim]> (<double complex*> params_buf[il])

                    rfft_mv = mkl_fft._numpy_fft.irfft(params_c_mv, axis=0)

                    pos_slice = pos_slice_buf_ptr[il]
                    n = 2*geodim*params_shapes[il,0]
                    dfac = 2*params_shapes[il,0]
                    scipy.linalg.cython_blas.daxpy(&n,&dfac,&rfft_mv[0,0],&int_one,pos_slice,&int_one)

            elif fft_backend == USE_DUCC_FFT:

                with gil:

                    rfft_mv = ducc0.fft.c2r(
                        np.asarray(<double complex[:(params_shapes[il,0]+1),:geodim]> (<double complex*> params_buf[il])),
                        axes=[0]                            ,
                        allow_overwriting_input = True      ,
                        lastsize = 2*params_shapes[il,0]    ,
                        forward = False                     ,
                        out = np.asarray(<double[:2*params_shapes[il,0],:geodim]> (pos_slice_buf_ptr[il])),
                    )

            elif fft_backend == USE_SCIPY_FFT:

                with gil:

                    params_c_mv = <double complex[:(params_shapes[il,0]+1),:geodim]> ( <double complex*> params_buf[il])

                    rfft_mv = scipy.fft.irfft(params_c_mv, axis=0, overwrite_x=True)

                    pos_slice = pos_slice_buf_ptr[il]
                    n = 2*geodim*params_shapes[il,0]
                    dfac = 2*params_shapes[il,0]
                    scipy.linalg.cython_blas.daxpy(&n,&dfac,&rfft_mv[0,0],&int_one,pos_slice,&int_one)

@cython.cdivision(True)
cdef void pos_slice_to_params(
    double** pos_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    double complex *params_basis_buf_ptr    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    Py_ssize_t* nnz_k_buf_ptr               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    double complex **ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             , int direction                         ,
    double **params_buf                     , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    int[::1] ParamBasisShortcut             ,
    int fft_backend                         , pyfftw.fftw_exe** fftw_genirfft_exe   , pyfftw.fftw_exe** fftw_symrfft_exe    ,
) noexcept nogil:

    cdef Py_ssize_t k, kmax, dk, ddk

    cdef double complex* ifft
    cdef double complex* params_basis
    cdef Py_ssize_t* nnz_k
    cdef double* pos_slice

    cdef int nloop = ncoeff_min_loop.shape[0]
    cdef Py_ssize_t il, i

    cdef int npr
    cdef int ncoeff_min_loop_il
    cdef int ncoeff_min_loop_nnz
    cdef int geodim = params_basis_shapes[0,0]
    cdef int nppl

    cdef double [:,:,::1] params_mv
    cdef double complex[:,:,::1] ifft_mv

    cdef double complex[:,::1] params_c_mv
    cdef double[:,::1] pos_slice_mv

    cdef int n
    cdef double* dest
    cdef double* src
    cdef double fac

    for il in range(nloop):

        if ParamBasisShortcut[il] == GENERAL_SYM:

            if params_basis_shapes[il,1] > 0:

                ifft = ifft_buf_ptr[il]
                params_basis = params_basis_buf_ptr + params_basis_shifts[il]
                nnz_k = nnz_k_buf_ptr + nnz_k_shifts[il]
                pos_slice = pos_slice_buf_ptr[il]

                npr = ifft_shapes[il,0] - 1
                ncoeff_min_loop_nnz = nnz_k_shapes[il,0]
                ncoeff_min_loop_il = ncoeff_min_loop[il]
                nppl = ifft_shapes[il,2] 

                if n_sub_fft[il] == 2:
                    pos_slice_to_partial_fft_2_sub(
                        pos_slice, params_basis, nnz_k, ifft,
                        npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                        direction,
                    )
                else:
                    pos_slice_to_partial_fft_1_sub(
                        pos_slice, params_basis, nnz_k, ifft,
                        npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                        direction,
                    )


            if fft_backend == USE_FFTW_FFT:

                if params_shapes[il,1] > 0:

                    pyfftw.execute_in_nogil(fftw_genirfft_exe[il])

                    dest = params_buf[il]

                    # Renormalization
                    n = (params_shifts[il+1] - params_shifts[il])
                    fac = 1. / (2*params_shapes[il,0])
                    scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)

            elif fft_backend == USE_MKL_FFT:

                with gil:

                    if params_shapes[il,1] > 0:

                        ifft_mv = <double complex[:ifft_shapes[il,0],:ifft_shapes[il,1],:ifft_shapes[il,2]:1]> ifft_buf_ptr[il]

                        params_mv = mkl_fft._numpy_fft.irfft(ifft_mv, axis=0)

                        dest = params_buf[il]
                        n = (params_shifts[il+1] - params_shifts[il])
                        scipy.linalg.cython_blas.dcopy(&n,&params_mv[0,0,0],&int_one,dest,&int_one)

            elif fft_backend == USE_DUCC_FFT:

                with gil:

                    if params_shapes[il,1] > 0:

                        params_mv = ducc0.fft.c2r(
                            np.asarray(<double complex[:ifft_shapes[il,0],:ifft_shapes[il,1],:ifft_shapes[il,2]:1]> (ifft_buf_ptr[il])),
                            axes=[0]                            ,
                            allow_overwriting_input = True      ,
                            lastsize = 2*ifft_shapes[il,0]-2    ,
                            forward = False                     ,
                            inorm = 2                           ,
                            out = np.asarray(<double[:2*params_shapes[il,0],:params_shapes[il,1],:params_shapes[il,2]:1]> (params_buf[il])),
                        )

            elif fft_backend == USE_SCIPY_FFT:

                with gil:

                    if params_shapes[il,1] > 0:

                        ifft_mv = <double complex[:ifft_shapes[il,0],:ifft_shapes[il,1],:ifft_shapes[il,2]:1]> ifft_buf_ptr[il]

                        params_mv = scipy.fft.irfft(ifft_mv, axis=0, overwrite_x=True)

                        dest = params_buf[il]
                        n = (params_shifts[il+1] - params_shifts[il])
                        scipy.linalg.cython_blas.dcopy(&n,&params_mv[0,0,0],&int_one,dest,&int_one)

            if direction < 0:

                dest = params_buf[il]

                if nnz_k_shapes[il,0] > 0:
                    if nnz_k_buf_ptr[nnz_k_shifts[il]] == 0:
                        for i in range(params_shapes[il,2]):
                            dest[i] *= 0.5

        elif ParamBasisShortcut[il] == RFFT:

            if fft_backend == USE_FFTW_FFT:

                pyfftw.execute_in_nogil(fftw_symrfft_exe[il])

                dest = params_buf[il]
                n = (params_shifts[il+1] - params_shifts[il])
                if direction > 0:
                    fac = 1. / (2*params_shapes[il,0])
                    scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)
                else:
                    fac = 2.
                    n -= 2*geodim
                    dest += 2*geodim
                    scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)

            elif fft_backend == USE_MKL_FFT:

                with gil:

                    n = 2*(ifft_shapes[il,0] - 1)
                    pos_slice_mv = <double[:n,:geodim:1]> (pos_slice_buf_ptr[il])

                    params_c_mv = mkl_fft._numpy_fft.rfft(pos_slice_mv, axis=0)

                    dest = params_buf[il]
                    src = <double*> &params_c_mv[0,0]
                    n = (params_shifts[il+1] - params_shifts[il])
                    scipy.linalg.cython_blas.dcopy(&n,src,&int_one,dest,&int_one)

                    if direction > 0:
                        fac = 1. / (2*params_shapes[il,0])
                        scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)
                    else:
                        fac = 2.
                        n -= 2*geodim
                        dest += 2*geodim
                        scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)

            elif fft_backend == USE_DUCC_FFT:

                with gil:

                    if direction > 0:

                        n = 2*(ifft_shapes[il,0] - 1)

                        params_c_mv = ducc0.fft.r2c(
                            np.asarray(<double[:n,:geodim:1]> (pos_slice_buf_ptr[il])),
                            axes=[0],
                            inorm = 2,
                            out = np.asarray(<double complex[:ifft_shapes[il,0],:geodim:1]> (<double complex*>params_buf[il])),
                        )

                    else:

                        n = 2*(ifft_shapes[il,0] - 1)

                        params_c_mv = ducc0.fft.r2c(
                            np.asarray(<double[:n,:geodim:1]> (pos_slice_buf_ptr[il])),
                            axes=[0],
                            out = np.asarray(<double complex[:ifft_shapes[il,0],:geodim:1]> (<double complex*>params_buf[il])),
                        )

                        dest = params_buf[il] + 2*geodim
                        src = <double*> &params_c_mv[0,0]
                        n = (params_shifts[il+1] - params_shifts[il]) - 2*geodim
                        fac = 2.
                        scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)

            elif fft_backend == USE_SCIPY_FFT:

                with gil:

                    n = 2*(ifft_shapes[il,0] - 1)
                    pos_slice_mv = <double[:n,:geodim:1]> (pos_slice_buf_ptr[il])

                    params_c_mv = scipy.fft.rfft(pos_slice_mv, axis=0, overwrite_x=True)

                    dest = params_buf[il]
                    src = <double*> &params_c_mv[0,0]
                    n = (params_shifts[il+1] - params_shifts[il])
                    scipy.linalg.cython_blas.dcopy(&n,src,&int_one,dest,&int_one)

                    if direction > 0:
                        fac = 1. / (2*params_shapes[il,0])
                        scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)
                    else:
                        fac = 2.
                        n -= 2*geodim
                        dest += 2*geodim
                        scipy.linalg.cython_blas.dscal(&n,&fac,dest,&int_one)

cdef void pos_slice_to_segmpos(
    double** pos_slice_buf_ptr          , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts  ,
    double* segmpos_buf_ptr             ,
    bint[::1] InterSpaceRotIsId         ,
    double[:,:,::1] InterSpaceRot       ,
    Py_ssize_t[::1] InterTimeRev        ,
    Py_ssize_t[::1] gensegm_to_body     ,
    Py_ssize_t[::1] gensegm_to_iint     ,
    Py_ssize_t[::1] BodyLoop            ,
    Py_ssize_t segm_size                ,
    Py_ssize_t segm_store               ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef double* pos_slice
    cdef double* segmpos
    cdef double* tmp_loc
    cdef double* tmp

    cdef int geodim = InterSpaceRot.shape[1]
    cdef int minus_geodim = -geodim
    cdef int segm_store_int = segm_store
    cdef int nitems_size = segm_size*geodim
    cdef int nitems_store = segm_store*geodim
    cdef Py_ssize_t isegm, ib, il, iint
    cdef Py_ssize_t i, idim

    cdef bint NeedsAllocate = False

    for isegm in range(nsegm):
        NeedsAllocate = (NeedsAllocate or ((InterTimeRev[isegm] < 0) and not(InterSpaceRotIsId[isegm])))

    if NeedsAllocate:
        tmp_loc = <double*> malloc(sizeof(double)*nitems_store)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]

        if InterTimeRev[isegm] > 0:

            pos_slice = pos_slice_buf_ptr[il] + nitems_size*iint
            segmpos = segmpos_buf_ptr + nitems_store*isegm

            if InterSpaceRotIsId[isegm]:
                scipy.linalg.cython_blas.dcopy(&nitems_store,pos_slice,&int_one,segmpos,&int_one)

            else:
                scipy.linalg.cython_blas.dgemm(transn, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, segmpos, &geodim)

        else:

            pos_slice = pos_slice_buf_ptr[il] + nitems_size*iint
            segmpos = segmpos_buf_ptr + nitems_store*(isegm+1) - geodim

            if InterSpaceRotIsId[isegm]:

                for i in range(segm_store):
                    for idim in range(geodim):
                        segmpos[idim] = pos_slice[idim]
                    segmpos -= geodim
                    pos_slice += geodim
                            
            else:
                tmp = tmp_loc
                scipy.linalg.cython_blas.dgemm(transn, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, tmp, &geodim)

                for i in range(segm_store):
                    for idim in range(geodim):
                        segmpos[idim] = tmp[idim]
                    segmpos -= geodim
                    tmp += geodim

    if NeedsAllocate:
        free(tmp_loc)

cdef void segmpos_to_pos_slice(
    double* segmpos_buf_ptr         ,
    double** pos_slice_buf_ptr      , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts  ,
    bint[::1] InterSpaceRotIsId     ,
    double[:,:,::1] InterSpaceRot   ,
    Py_ssize_t[::1] InterTimeRev    ,
    Py_ssize_t[::1] gensegm_to_body ,
    Py_ssize_t[::1] gensegm_to_iint ,
    Py_ssize_t[::1] BodyLoop        ,
    Py_ssize_t segm_size            ,
    Py_ssize_t segm_store           ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef double* pos_slice
    cdef double* segmpos
    cdef double* tmp_loc
    cdef double* tmp

    cdef int geodim = InterSpaceRot.shape[1]
    cdef int segm_store_int = segm_store
    cdef int nitems_size = segm_size*geodim
    cdef int nitems_store = segm_store*geodim
    cdef Py_ssize_t isegm, ib, il, iint
    cdef Py_ssize_t i, idim

    cdef bint NeedsAllocate = False

    for isegm in range(nsegm):
        NeedsAllocate = (NeedsAllocate or ((InterTimeRev[isegm] < 0) and not(InterSpaceRotIsId[isegm])))

    if NeedsAllocate:
        tmp_loc = <double*> malloc(sizeof(double)*nitems_store)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]

        if InterTimeRev[isegm] > 0:

            pos_slice = pos_slice_buf_ptr[il] + nitems_size*iint
            segmpos = segmpos_buf_ptr + nitems_store*isegm

            if InterSpaceRotIsId[isegm]:
                scipy.linalg.cython_blas.dcopy(&nitems_store,segmpos,&int_one,pos_slice,&int_one)
            else:
                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, segmpos, &geodim, &zero_double, pos_slice, &geodim)

        else:

            pos_slice = pos_slice_buf_ptr[il] + nitems_size*iint

            if InterSpaceRotIsId[isegm]:

                segmpos = segmpos_buf_ptr + nitems_store*(isegm+1) - geodim

                for i in range(segm_store):
                    for idim in range(geodim):
                        pos_slice[idim] = segmpos[idim]
                    segmpos -= geodim
                    pos_slice += geodim
                            
            else:

                segmpos = segmpos_buf_ptr + nitems_store*(isegm)
                tmp = tmp_loc

                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, segmpos, &geodim, &zero_double, tmp, &geodim)

                tmp = tmp_loc + nitems_store - geodim
                for i in range(segm_store):
                    for idim in range(geodim):
                        pos_slice[idim] = tmp[idim]
                    pos_slice += geodim
                    tmp -= geodim

    if NeedsAllocate:
        free(tmp_loc)

cdef void segmpos_to_pos_slice_T(
    double* segmpos_buf_ptr         ,
    double** pos_slice_buf_ptr      , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts  ,
    bint[::1] InterSpaceRotIsId     ,
    double[:,:,::1] InterSpaceRot   ,
    Py_ssize_t[::1] InterTimeRev    ,
    Py_ssize_t[::1] gensegm_to_body ,
    Py_ssize_t[::1] gensegm_to_iint ,
    Py_ssize_t[::1] BodyLoop        ,
    Py_ssize_t segm_size            ,
    Py_ssize_t segm_store           ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef double* pos_slice
    cdef double* segmpos
    cdef double* tmp_loc
    cdef double* tmp

    cdef int geodim = InterSpaceRot.shape[1]
    cdef int segm_store_int = segm_store
    cdef int nitems_size = segm_size*geodim
    cdef int nitems_store = segm_store*geodim
    cdef Py_ssize_t isegm, ib, il, iint
    cdef Py_ssize_t i, idim

    cdef bint NeedsAllocate = False

    for isegm in range(nsegm):
        NeedsAllocate = (NeedsAllocate or ((InterTimeRev[isegm] < 0) and not(InterSpaceRotIsId[isegm])))

    if NeedsAllocate:
        tmp_loc = <double*> malloc(sizeof(double)*nitems_store)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]

        if InterTimeRev[isegm] > 0:

            pos_slice = pos_slice_buf_ptr[il] + nitems_size*iint
            segmpos = segmpos_buf_ptr + nitems_store*isegm

            if InterSpaceRotIsId[isegm]:
                scipy.linalg.cython_blas.daxpy(&nitems_store, &one_double, segmpos,&int_one,pos_slice,&int_one)

            else:
                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, segmpos, &geodim, &one_double, pos_slice, &geodim)

        else:
    
            pos_slice = pos_slice_buf_ptr[il] + nitems_size*iint

            if InterSpaceRotIsId[isegm]:

                segmpos = segmpos_buf_ptr + nitems_store*(isegm+1) - geodim

                for i in range(segm_store):
                    for idim in range(geodim):
                        pos_slice[idim] += segmpos[idim]
                    segmpos -= geodim
                    pos_slice += geodim
                            
            else:

                segmpos = segmpos_buf_ptr + nitems_store*(isegm)
                tmp = tmp_loc

                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, segmpos, &geodim, &zero_double, tmp, &geodim)

                tmp = tmp_loc + nitems_store - geodim
                for i in range(segm_store):
                    for idim in range(geodim):
                        pos_slice[idim] += tmp[idim]
                    pos_slice += geodim
                    tmp -= geodim

    if NeedsAllocate:
        free(tmp_loc)

@cython.cdivision(True)
cdef void params_to_segmpos(
    double[::1] params_mom_buf              ,
    double** params_pos_buf                 , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    double complex** ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    double complex[::1] params_basis_buf    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    Py_ssize_t[::1] nnz_k_buf               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    bint[::1] co_in_buf                     , Py_ssize_t[:,::1] co_in_shapes        , Py_ssize_t[::1] co_in_shifts          ,
    double** pos_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             , int fft_backend                       ,
    int[::1] ParamBasisShortcutPos          ,
    pyfftw.fftw_exe** fftw_genrfft_exe      , pyfftw.fftw_exe** fftw_symirfft_exe   ,
    Py_ssize_t[::1] loopnb                  , double[::1] loopmass                  ,
    bint[::1] InterSpaceRotIsId             , double[:,:,::1] InterSpaceRot         , Py_ssize_t[::1] InterTimeRev          ,
    Py_ssize_t[::1] ALG_Iint                , double[:,:,::1] ALG_SpaceRot          , Py_ssize_t[::1] ALG_TimeRev           ,
    Py_ssize_t[::1] gensegm_to_body         ,
    Py_ssize_t[::1] gensegm_to_iint         ,
    Py_ssize_t[::1] BodyLoop                ,
    Py_ssize_t segm_size                    ,
    Py_ssize_t segm_store                   ,
    double[:,:,::1] segmpos                 ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef int geodim = InterSpaceRot.shape[1]

    changevar_mom_pos(
        &params_mom_buf[0]  , params_shapes , params_shifts ,
        nnz_k_buf           , nnz_k_shapes  , nnz_k_shifts  ,
        co_in_buf           , co_in_shapes  , co_in_shifts  ,
        ncoeff_min_loop     ,
        loopnb              , loopmass      ,
        params_pos_buf      , 
    )   

    params_to_pos_slice(
        params_pos_buf          , params_shapes         , params_shifts         ,
        &nnz_k_buf[0]           , nnz_k_shapes          , nnz_k_shifts          ,
        ifft_buf_ptr            , ifft_shapes           , ifft_shifts           ,
        ParamBasisShortcutPos   ,
        fft_backend             , fftw_genrfft_exe      , fftw_symirfft_exe     ,
        &params_basis_buf[0]    , params_basis_shapes   , params_basis_shifts   ,
        pos_slice_buf_ptr       , pos_slice_shapes      , pos_slice_shifts      ,
        ncoeff_min_loop         , n_sub_fft             ,
    )

    if (segm_size != segm_store):

        Adjust_after_last_gen(
            pos_slice_buf_ptr   , pos_slice_shifts  ,
            ifft_shapes         ,
            params_basis_shapes ,
            n_sub_fft           ,
            ALG_Iint            ,
            ALG_TimeRev         , ALG_SpaceRot      ,
            segm_size           ,
        )

    pos_slice_to_segmpos(
        pos_slice_buf_ptr   , pos_slice_shapes  , pos_slice_shifts ,
        &segmpos[0,0,0]     ,
        InterSpaceRotIsId   ,
        InterSpaceRot       ,
        InterTimeRev        ,
        gensegm_to_body     ,
        gensegm_to_iint     ,
        BodyLoop            ,
        segm_size           ,
        segm_store          ,
    )

@cython.cdivision(True)
cdef void params_to_segmvel(
    double[::1] params_mom_buf              ,
    double** params_vel_buf                 , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    double complex** ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    double complex[::1] params_basis_buf    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    Py_ssize_t[::1] nnz_k_buf               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    bint[::1] co_in_buf                     , Py_ssize_t[:,::1] co_in_shapes        , Py_ssize_t[::1] co_in_shifts          ,
    double** vel_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             , int fft_backend                       ,
    int[::1] ParamBasisShortcutVel          ,
    pyfftw.fftw_exe** fftw_genrfft_exe      , pyfftw.fftw_exe** fftw_symirfft_exe   ,
    Py_ssize_t[::1] loopnb                  , double[::1] loopmass                  ,
    bint[::1] InterSpaceRotIsId             , double[:,:,::1] InterSpaceRot         , Py_ssize_t[::1] InterTimeRev          ,
    Py_ssize_t[::1] ALG_Iint                , double[:,:,::1] ALG_SpaceRot          , Py_ssize_t[::1] ALG_TimeRev           ,
    Py_ssize_t[::1] gensegm_to_body         ,
    Py_ssize_t[::1] gensegm_to_iint         ,
    Py_ssize_t[::1] BodyLoop                ,
    Py_ssize_t segm_size                    ,
    Py_ssize_t segm_store                   ,
    double[:,:,::1] segmvel                 ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef int geodim = InterSpaceRot.shape[1]

    changevar_mom_vel(
        &params_mom_buf[0]  , params_shapes , params_shifts ,
        nnz_k_buf           , nnz_k_shapes  , nnz_k_shifts  ,
        co_in_buf           , co_in_shapes  , co_in_shifts  ,
        ncoeff_min_loop     ,
        loopnb              , loopmass      ,
        params_vel_buf      , 
    )   

    params_to_pos_slice(
        params_vel_buf          , params_shapes         , params_shifts         ,
        &nnz_k_buf[0]           , nnz_k_shapes          , nnz_k_shifts          ,
        ifft_buf_ptr            , ifft_shapes           , ifft_shifts           ,
        ParamBasisShortcutVel   ,
        fft_backend             , fftw_genrfft_exe      , fftw_symirfft_exe     ,
        &params_basis_buf[0]    , params_basis_shapes   , params_basis_shifts   ,
        vel_slice_buf_ptr       , pos_slice_shapes      , pos_slice_shifts      ,
        ncoeff_min_loop         , n_sub_fft             ,
    )

    if (segm_size != segm_store):

        Adjust_after_last_gen(
            vel_slice_buf_ptr   , pos_slice_shifts      ,
            ifft_shapes         ,
            params_basis_shapes ,
            n_sub_fft           ,
            ALG_Iint            ,
            ALG_TimeRev         , ALG_SpaceRot          ,
            segm_size           ,
        )

    pos_slice_to_segmpos( 
        vel_slice_buf_ptr   , pos_slice_shapes  , pos_slice_shifts ,
        &segmvel[0,0,0]     ,
        InterSpaceRotIsId   ,
        InterSpaceRot       ,
        InterTimeRev        ,
        gensegm_to_body     ,
        gensegm_to_iint     ,
        BodyLoop            ,
        segm_size           ,
        segm_store          ,
    )

@cython.cdivision(True)
cdef void segmpos_to_params(
    double[:,:,::1] segmpos                 ,
    double** params_pos_buf                 , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    double complex **ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    double complex[::1] params_basis_buf    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    Py_ssize_t[::1] nnz_k_buf               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    bint[::1] co_in_buf                     , Py_ssize_t[:,::1] co_in_shapes        , Py_ssize_t[::1] co_in_shifts          ,
    double** pos_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             , int fft_backend                       ,
    int[::1] ParamBasisShortcutPos          ,
    pyfftw.fftw_exe** fftw_genirfft_exe     , pyfftw.fftw_exe** fftw_symrfft_exe                                            ,
    Py_ssize_t[::1] loopnb                  , double[::1] loopmass                  ,
    bint[::1] InterSpaceRotIsId             , double[:,:,::1] InterSpaceRot         , Py_ssize_t[::1] InterTimeRev          ,
    Py_ssize_t[::1] gensegm_to_body         ,
    Py_ssize_t[::1] gensegm_to_iint         ,
    Py_ssize_t[::1] BodyLoop                ,
    Py_ssize_t segm_size                    ,
    Py_ssize_t segm_store                   ,
    double[::1] params_mom_buf              ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef int geodim = InterSpaceRot.shape[1]

    segmpos_to_pos_slice(
        &segmpos[0,0,0]     ,
        pos_slice_buf_ptr   , pos_slice_shapes  , pos_slice_shifts ,
        InterSpaceRotIsId   ,
        InterSpaceRot       ,
        InterTimeRev        ,
        gensegm_to_body     ,
        gensegm_to_iint     ,
        BodyLoop            ,
        segm_size           ,
        segm_store          ,
    )

    pos_slice_to_params(
        pos_slice_buf_ptr       , pos_slice_shapes      , pos_slice_shifts      ,
        &params_basis_buf[0]    , params_basis_shapes   , params_basis_shifts   ,
        &nnz_k_buf[0]           , nnz_k_shapes          , nnz_k_shifts          ,
        ifft_buf_ptr            , ifft_shapes           , ifft_shifts           ,
        ncoeff_min_loop         , n_sub_fft             , 1                     ,
        params_pos_buf          , params_shapes         , params_shifts         ,
        ParamBasisShortcutPos   ,
        fft_backend             , fftw_genirfft_exe     , fftw_symrfft_exe      ,
    )

    changevar_mom_pos_inv(
        params_pos_buf      , params_shapes , params_shifts ,
        nnz_k_buf           , nnz_k_shapes  , nnz_k_shifts  ,
        co_in_buf           , co_in_shapes  , co_in_shifts  ,
        ncoeff_min_loop     ,
        loopnb              , loopmass      ,
        &params_mom_buf[0]  , 
    )   

@cython.cdivision(True)
cdef void segmpos_to_params_T(
    double[:,:,::1] segmpos                 ,
    double** params_pos_buf                 , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    double complex **ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    double complex[::1] params_basis_buf    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    Py_ssize_t[::1] nnz_k_buf               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    bint[::1] co_in_buf                     , Py_ssize_t[:,::1] co_in_shapes        , Py_ssize_t[::1] co_in_shifts          ,
    double** pos_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             , int fft_backend                       ,
    int[::1] ParamBasisShortcutPos          ,
    pyfftw.fftw_exe** fftw_genirfft_exe     , pyfftw.fftw_exe** fftw_symrfft_exe                                            ,
    Py_ssize_t[::1] loopnb                  , double[::1] loopmass                  ,
    bint[::1] InterSpaceRotIsId             , double[:,:,::1] InterSpaceRot         , Py_ssize_t[::1] InterTimeRev          ,
    Py_ssize_t[::1] ALG_Iint                , double[:,:,::1] ALG_SpaceRot          , Py_ssize_t[::1] ALG_TimeRev           ,
    Py_ssize_t[::1] gensegm_to_body         ,
    Py_ssize_t[::1] gensegm_to_iint         ,
    Py_ssize_t[::1] BodyLoop                ,
    Py_ssize_t segm_size                    ,
    Py_ssize_t segm_store                   ,
    double[::1] params_mom_buf              ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef int geodim = InterSpaceRot.shape[1]
    cdef Py_ssize_t il
    cdef Py_ssize_t nloop = ncoeff_min_loop.shape[0]

    for il in range(nloop):
        memset(pos_slice_buf_ptr[il], 0, sizeof(double)*(pos_slice_shifts[il+1]-pos_slice_shifts[il]))

    segmpos_to_pos_slice_T(
        &segmpos[0,0,0]     ,
        pos_slice_buf_ptr   , pos_slice_shapes  , pos_slice_shifts ,
        InterSpaceRotIsId   ,
        InterSpaceRot       ,
        InterTimeRev        ,
        gensegm_to_body     ,
        gensegm_to_iint     ,
        BodyLoop            ,
        segm_size           ,
        segm_store          ,
    )

    if (segm_size != segm_store):
        Adjust_after_last_gen_T(
            pos_slice_buf_ptr   , pos_slice_shifts  ,
            ifft_shapes         ,
            params_basis_shapes ,
            n_sub_fft           ,
            ALG_Iint            ,
            ALG_TimeRev         , ALG_SpaceRot      ,
            segm_size           ,
        )

    pos_slice_to_params(
        pos_slice_buf_ptr       , pos_slice_shapes      , pos_slice_shifts      ,
        &params_basis_buf[0]    , params_basis_shapes   , params_basis_shifts   ,
        &nnz_k_buf[0]           , nnz_k_shapes          , nnz_k_shifts          ,
        ifft_buf_ptr            , ifft_shapes           , ifft_shifts           ,
        ncoeff_min_loop         , n_sub_fft             , -1                    ,
        params_pos_buf          , params_shapes         , params_shifts         ,
        ParamBasisShortcutPos   ,
        fft_backend             , fftw_genirfft_exe     , fftw_symrfft_exe      ,
    )

    changevar_mom_pos_T(
        params_pos_buf      , params_shapes , params_shifts ,
        nnz_k_buf           , nnz_k_shapes  , nnz_k_shifts  ,
        co_in_buf           , co_in_shapes  , co_in_shifts  ,
        ncoeff_min_loop     ,
        loopnb              , loopmass      ,
        &params_mom_buf[0]  , 
    )   

cdef inline int get_inter_flags(
    Py_ssize_t segm_size            , Py_ssize_t segm_store ,
    Py_ssize_t geodim               ,
    inter_law_fun_type inter_law
) noexcept nogil:

    cdef int inter_flags = 0

    if (segm_size != segm_store):
        inter_flags += 1
    if (geodim == 2):
        inter_flags += 2
    if (inter_law == gravity_pot):
        inter_flags += 4

    return inter_flags

@cython.cdivision(True)
cdef void segm_pos_to_hash(
    double[:,:,::1] segmpos         ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         ,
    double[::1] Hash_exp            , double[::1] Hash              ,           
) noexcept nogil:

    cdef Py_ssize_t nbin = BinSourceSegm.shape[0]
    cdef int geodim = BinSpaceRot.shape[1]
    cdef int segm_store_int = segm_store
    cdef int nitems_int = segm_store*geodim
    cdef int nexp = Hash_exp.shape[0]
    cdef Py_ssize_t iexp
    cdef Py_ssize_t ibin, idim
    cdef Py_ssize_t isegm, isegmp

    cdef double pot
    cdef double dx2, a
    cdef double bin_fac
    cdef double* hash_tmp = <double*> malloc(sizeof(double)*nexp)

    cdef bint size_is_store = (segm_size == segm_store)

    cdef double* tmp_loc_dpos = <double*> malloc(sizeof(double)*segm_store*geodim)
    cdef double* dpos

    for iexp in range(nexp):
        Hash[iexp] = 0

    for ibin in range(nbin):

        dpos = tmp_loc_dpos

        isegm = BinSourceSegm[ibin]
        isegmp = BinTargetSegm[ibin]

        if BinSpaceRotIsId[ibin]:
            scipy.linalg.cython_blas.dcopy(&nitems_int, &segmpos[isegm,0,0], &int_one, tmp_loc_dpos, &int_one)

        else:
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &BinSpaceRot[ibin,0,0], &geodim, &segmpos[isegm,0,0], &geodim, &zero_double, tmp_loc_dpos, &geodim)

        scipy.linalg.cython_blas.daxpy(&nitems_int, &minusone_double, &segmpos[isegmp,0,0], &int_one, tmp_loc_dpos, &int_one)

        bin_fac = BinProdChargeSum[ibin]
        bin_fac /= segm_size

        memset(hash_tmp, 0, sizeof(double)*nexp)

        if size_is_store:

            for iint in range(segm_size):

                dx2 = dpos[0]*dpos[0]
                for idim in range(1,geodim):
                    dx2 += dpos[idim]*dpos[idim]

                for iexp in range(nexp):
                    hash_tmp[iexp] += cpow(dx2, Hash_exp[iexp])

                dpos += geodim

        else:
            
            # First iteration
            dx2 = dpos[0]*dpos[0]
            for idim in range(1,geodim):
                dx2 += dpos[idim]*dpos[idim]

            for iexp in range(nexp):
                hash_tmp[iexp] += 0.5 * cpow(dx2, Hash_exp[iexp])

            dpos += geodim

            for iint in range(1,segm_size):

                dx2 = dpos[0]*dpos[0]
                for idim in range(1,geodim):
                    dx2 += dpos[idim]*dpos[idim]

                for iexp in range(nexp):
                    hash_tmp[iexp] += cpow(dx2, Hash_exp[iexp])

                dpos += geodim

            # Last iteration
            dx2 = dpos[0]*dpos[0]
            for idim in range(1,geodim):
                dx2 += dpos[idim]*dpos[idim]

            for iexp in range(nexp):
                hash_tmp[iexp] += 0.5 * cpow(dx2, Hash_exp[iexp])

        scipy.linalg.cython_blas.daxpy(&nexp, &bin_fac, hash_tmp, &int_one, &Hash[0], &int_one)

    free(hash_tmp)
    free(tmp_loc_dpos)

@cython.cdivision(True)
cdef double segm_pos_to_pot_nrg(
    double[:,:,::1] segmpos         ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr     ,
) noexcept nogil:

    cdef Py_ssize_t nbin = BinSourceSegm.shape[0]
    cdef int geodim = BinSpaceRot.shape[1]
    cdef Py_ssize_t geodim_size = geodim
    cdef int segm_store_int = segm_store
    cdef int nitems_int = segm_store*geodim
    cdef Py_ssize_t ibin, idim
    cdef Py_ssize_t isegm, isegmp

    cdef double pot_nrg = 0.
    cdef double pot_nrg_bin
    cdef double dx2, a
    cdef double bin_fac

    cdef bint size_is_store = (segm_size == segm_store)

    cdef double* tmp_loc_dpos = <double*> malloc(sizeof(double)*segm_store*geodim)
    cdef double* dpos

    cdef double[3] pot

    for ibin in range(nbin):

        dpos = tmp_loc_dpos

        isegm = BinSourceSegm[ibin]
        isegmp = BinTargetSegm[ibin]

        if BinSpaceRotIsId[ibin]:
            scipy.linalg.cython_blas.dcopy(&nitems_int, &segmpos[isegm,0,0], &int_one, tmp_loc_dpos, &int_one)
        else:
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &BinSpaceRot[ibin,0,0], &geodim, &segmpos[isegm,0,0], &geodim, &zero_double, tmp_loc_dpos, &geodim)

        scipy.linalg.cython_blas.daxpy(&nitems_int, &minusone_double, &segmpos[isegmp,0,0], &int_one, tmp_loc_dpos, &int_one)

        pot_nrg_bin = 0

        if size_is_store:

            for iint in range(segm_size):

                dx2 = dpos[0]*dpos[0]
                for idim in range(1,geodim):
                    dx2 += dpos[idim]*dpos[idim]

                inter_law(dx2, pot, inter_law_param_ptr)

                pot_nrg_bin += pot[0]
                dpos += geodim

        else:
            
            # First iteration
            dx2 = dpos[0]*dpos[0]
            for idim in range(1,geodim):
                dx2 += dpos[idim]*dpos[idim]

            inter_law(dx2, pot, inter_law_param_ptr)

            pot_nrg_bin += 0.5*pot[0]
            dpos += geodim

            for iint in range(1,segm_size):

                dx2 = dpos[0]*dpos[0]
                for idim in range(1,geodim):
                    dx2 += dpos[idim]*dpos[idim]

                inter_law(dx2, pot, inter_law_param_ptr)

                pot_nrg_bin += pot[0]
                dpos += geodim

            # Last iteration
            dx2 = dpos[0]*dpos[0]
            for idim in range(1,geodim):
                dx2 += dpos[idim]*dpos[idim]

            inter_law(dx2, pot, inter_law_param_ptr)

            pot_nrg_bin += 0.5*pot[0]

        bin_fac = BinProdChargeSum[ibin]
        bin_fac /= segm_size

        pot_nrg += pot_nrg_bin * bin_fac

    free(tmp_loc_dpos)

    return pot_nrg

@cython.cdivision(True)
cdef void segm_pos_to_pot_nrg_grad(
    double[:,:,::1] segmpos         , double[:,:,::1] pot_nrg_grad  ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         , double globalmul  ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr     ,
) noexcept nogil:

    cdef Py_ssize_t nbin = BinSourceSegm.shape[0]
    cdef int geodim = BinSpaceRot.shape[1]
    cdef Py_ssize_t geodim_size = geodim
    cdef int segm_store_int = segm_store
    cdef int nitems_int = segm_store*geodim
    cdef Py_ssize_t nitems = sizeof(double)*segm_store*geodim
    cdef Py_ssize_t ibin, idim
    cdef Py_ssize_t isegm, isegmp

    cdef int inter_flags = get_inter_flags(
        segm_size   , segm_store    ,
        geodim_size , inter_law     ,
    )

    cdef double dx2
    cdef double bin_fac
    cdef double* dx = <double*> malloc(sizeof(double)*geodim)

    cdef double* tmp_loc_dpos = <double*> malloc(nitems)
    cdef double* tmp_loc_grad = <double*> malloc(nitems)

    for ibin in range(nbin):

        isegm = BinSourceSegm[ibin]
        isegmp = BinTargetSegm[ibin]

        if BinSpaceRotIsId[ibin]:
            scipy.linalg.cython_blas.dcopy(&nitems_int, &segmpos[isegm,0,0], &int_one, tmp_loc_dpos, &int_one)
        else:
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &BinSpaceRot[ibin,0,0], &geodim, &segmpos[isegm,0,0], &geodim, &zero_double, tmp_loc_dpos, &geodim)

        scipy.linalg.cython_blas.daxpy(&nitems_int, &minusone_double, &segmpos[isegmp,0,0], &int_one, tmp_loc_dpos, &int_one)

        pot_nrg_grad_inter(
                inter_flags     , segm_size             , geodim_size   ,      
                tmp_loc_dpos    , tmp_loc_grad          ,
                inter_law       , inter_law_param_ptr   ,
            )

        bin_fac = 2*BinProdChargeSum[ibin]*globalmul
        bin_fac /= segm_size

        if BinSpaceRotIsId[ibin]:
            scipy.linalg.cython_blas.daxpy(&nitems_int, &bin_fac, tmp_loc_grad, &int_one, &pot_nrg_grad[isegm,0,0], &int_one)
        else:
            scipy.linalg.cython_blas.dgemm(transn, transn, &geodim, &segm_store_int, &geodim, &bin_fac, &BinSpaceRot[ibin,0,0], &geodim, tmp_loc_grad, &geodim, &one_double, &pot_nrg_grad[isegm,0,0], &geodim)

        bin_fac = -bin_fac
        scipy.linalg.cython_blas.daxpy(&nitems_int, &bin_fac, tmp_loc_grad, &int_one, &pot_nrg_grad[isegmp,0,0], &int_one)

    free(dx)
    free(tmp_loc_grad)
    free(tmp_loc_dpos)

@cython.cdivision(True)
cdef void pot_nrg_grad_inter(
    int inter_flags , Py_ssize_t segm_size  , Py_ssize_t geodim         ,      
    double* dpos_in , double* grad_in       ,
    inter_law_fun_type inter_law            , void* inter_law_param_ptr ,
) noexcept nogil:

    if inter_flags == 0:

        pot_nrg_grad_inter_size_law_nd(
            segm_size   , geodim                ,      
            dpos_in     , grad_in               ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 1:

        pot_nrg_grad_inter_store_law_nd(
            segm_size   , geodim                ,      
            dpos_in     , grad_in               ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 2:

        pot_nrg_grad_inter_size_law_2d(
            segm_size   ,      
            dpos_in     , grad_in               ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 3:

        pot_nrg_grad_inter_store_law_2d(
            segm_size   ,      
            dpos_in     , grad_in               ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 4:

        pot_nrg_grad_inter_size_gravity_nd(
            segm_size   , geodim    ,      
            dpos_in     , grad_in   ,
        )

    elif inter_flags == 5:

        pot_nrg_grad_inter_store_gravity_nd(
            segm_size   , geodim    ,      
            dpos_in     , grad_in   ,
        )

    elif inter_flags == 6:

        pot_nrg_grad_inter_size_gravity_2d(
            segm_size   ,      
            dpos_in     , grad_in   ,
        )

    elif inter_flags == 7:

        pot_nrg_grad_inter_store_gravity_2d(
            segm_size   ,      
            dpos_in     , grad_in   ,
        )

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_size_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    for iint in range(segm_size):

        dx2 = dpos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += dpos[idim]*dpos[idim]

        inter_law(dx2, pot, inter_law_param_ptr)

        for idim in range(geodim):
            grad[idim] = pot[1]*dpos[idim]

        dpos += geodim
        grad += geodim

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_store_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    # First iteration
    dx2 = dpos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += dpos[idim]*dpos[idim]

    inter_law(dx2, pot, inter_law_param_ptr)

    for idim in range(geodim):
        grad[idim] = 0.5*pot[1]*dpos[idim]

    dpos += geodim
    grad += geodim

    for iint in range(1,segm_size):

        dx2 = dpos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += dpos[idim]*dpos[idim]

        inter_law(dx2, pot, inter_law_param_ptr)

        for idim in range(geodim):
            grad[idim] = pot[1]*dpos[idim]

        dpos += geodim
        grad += geodim

    # Last iteration
    dx2 = dpos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += dpos[idim]*dpos[idim]

    inter_law(dx2, pot, inter_law_param_ptr)

    for idim in range(geodim):
        grad[idim] = 0.5*pot[1]*dpos[idim]

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_size_law_2d(
    Py_ssize_t segm_size            ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    for iint in range(segm_size):

        dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

        inter_law(dx2, pot, inter_law_param_ptr)

        grad[0] = pot[1]*dpos[0]
        grad[1] = pot[1]*dpos[1]

        dpos += 2
        grad += 2

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_store_law_2d(
    Py_ssize_t segm_size            ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    # First iteration
    dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

    inter_law(dx2, pot, inter_law_param_ptr)

    grad[0] = 0.5*pot[1]*dpos[0]
    grad[1] = 0.5*pot[1]*dpos[1]

    dpos += 2
    grad += 2

    for iint in range(1,segm_size):

        dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

        inter_law(dx2, pot, inter_law_param_ptr)

        grad[0] = pot[1]*dpos[0]
        grad[1] = pot[1]*dpos[1]

        dpos += 2
        grad += 2

    # Last iteration
    dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

    inter_law(dx2, pot, inter_law_param_ptr)

    grad[0] = 0.5*pot[1]*dpos[0]
    grad[1] = 0.5*pot[1]*dpos[1]

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_size_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,        
    double* dpos_in         , double* grad_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    for iint in range(segm_size):

        dx2 = dpos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += dpos[idim]*dpos[idim]

        inline_gravity_pot(dx2, pot)

        for idim in range(geodim):
            grad[idim] = pot[1]*dpos[idim]

        dpos += geodim
        grad += geodim

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_store_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,      
    double* dpos_in         , double* grad_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    # First iteration
    dx2 = dpos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += dpos[idim]*dpos[idim]

    inline_gravity_pot(dx2, pot)

    for idim in range(geodim):
        grad[idim] = 0.5*pot[1]*dpos[idim]

    dpos += geodim
    grad += geodim

    for iint in range(1,segm_size):

        dx2 = dpos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += dpos[idim]*dpos[idim]

        inline_gravity_pot(dx2, pot)

        for idim in range(geodim):
            grad[idim] = pot[1]*dpos[idim]

        dpos += geodim
        grad += geodim

    # Last iteration
    dx2 = dpos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += dpos[idim]*dpos[idim]

    inline_gravity_pot(dx2, pot)

    for idim in range(geodim):
        grad[idim] = 0.5*pot[1]*dpos[idim]

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_size_gravity_2d(
    Py_ssize_t segm_size    ,      
    double* dpos_in         , double* grad_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    for iint in range(segm_size):

        dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

        inline_gravity_pot(dx2, pot)

        grad[0] = pot[1]*dpos[0]
        grad[1] = pot[1]*dpos[1]

        dpos += 2
        grad += 2

@cython.cdivision(True)
cdef void pot_nrg_grad_inter_store_gravity_2d(
    Py_ssize_t segm_size    ,      
    double* dpos_in         , double* grad_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2
    cdef double[3] pot

    cdef double* dpos = dpos_in
    cdef double* grad = grad_in

    # First iteration
    dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

    inline_gravity_pot(dx2, pot)

    grad[0] = 0.5*pot[1]*dpos[0]
    grad[1] = 0.5*pot[1]*dpos[1]

    dpos += 2
    grad += 2

    for iint in range(1,segm_size):

        dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

        inline_gravity_pot(dx2, pot)

        grad[0] = pot[1]*dpos[0]
        grad[1] = pot[1]*dpos[1]

        dpos += 2
        grad += 2

    # Last iteration
    dx2 = dpos[0]*dpos[0] + dpos[1]*dpos[1]

    inline_gravity_pot(dx2, pot)

    grad[0] = 0.5*pot[1]*dpos[0]
    grad[1] = 0.5*pot[1]*dpos[1]

@cython.cdivision(True)
cdef void segm_pos_to_pot_nrg_hess(
    double[:,:,::1] segmpos         , double[:,:,::1] dsegmpos      , double[:,:,::1] pot_nrg_hess  ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         , double globalmul              ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr     ,
) noexcept nogil:

    cdef Py_ssize_t nbin = BinSourceSegm.shape[0]
    cdef int geodim = BinSpaceRot.shape[1]
    cdef Py_ssize_t geodim_size = geodim
    cdef int segm_store_int = segm_store
    cdef int nitems_int = segm_store*geodim
    cdef Py_ssize_t nitems = sizeof(double)*segm_store*geodim
    cdef Py_ssize_t ibin, idim
    cdef Py_ssize_t isegm, isegmp

    cdef int inter_flags = get_inter_flags(
        segm_size   , segm_store    ,
        geodim_size , inter_law     ,
    )

    cdef double bin_fac
    cdef double* tmp_loc_pos = <double*> malloc(nitems)
    cdef double* tmp_loc_dpos = <double*> malloc(nitems)
    cdef double* tmp_loc_hess = <double*> malloc(nitems)

    for ibin in range(nbin):

        isegm = BinSourceSegm[ibin]
        isegmp = BinTargetSegm[ibin]

        if BinSpaceRotIsId[ibin]:
            scipy.linalg.cython_blas.dcopy(&nitems_int, &segmpos[isegm,0,0], &int_one, tmp_loc_pos, &int_one)
            scipy.linalg.cython_blas.dcopy(&nitems_int, &dsegmpos[isegm,0,0], &int_one, tmp_loc_dpos, &int_one)

        else:
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &BinSpaceRot[ibin,0,0], &geodim, &segmpos[isegm,0,0], &geodim, &zero_double, tmp_loc_pos, &geodim)
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &BinSpaceRot[ibin,0,0], &geodim, &dsegmpos[isegm,0,0], &geodim, &zero_double, tmp_loc_dpos, &geodim)

        scipy.linalg.cython_blas.daxpy(&nitems_int, &minusone_double, &segmpos[isegmp,0,0], &int_one, tmp_loc_pos, &int_one)
        scipy.linalg.cython_blas.daxpy(&nitems_int, &minusone_double, &dsegmpos[isegmp,0,0], &int_one, tmp_loc_dpos, &int_one)

        pot_nrg_hess_inter(
                inter_flags     , segm_size     , geodim_size   ,      
                tmp_loc_pos     , tmp_loc_dpos  , tmp_loc_hess  ,
                inter_law       , inter_law_param_ptr           ,
            )

        bin_fac = 2*BinProdChargeSum[ibin]*globalmul
        bin_fac /= segm_size

        if BinSpaceRotIsId[ibin]:
            scipy.linalg.cython_blas.daxpy(&nitems_int, &bin_fac, tmp_loc_hess, &int_one, &pot_nrg_hess[isegm,0,0], &int_one)
        else:
            scipy.linalg.cython_blas.dgemm(transn, transn, &geodim, &segm_store_int, &geodim, &bin_fac, &BinSpaceRot[ibin,0,0], &geodim, tmp_loc_hess, &geodim, &one_double, &pot_nrg_hess[isegm,0,0], &geodim)

        bin_fac = -bin_fac
        scipy.linalg.cython_blas.daxpy(&nitems_int, &bin_fac, tmp_loc_hess, &int_one, &pot_nrg_hess[isegmp,0,0], &int_one)

    free(tmp_loc_hess)
    free(tmp_loc_pos)
    free(tmp_loc_dpos)

@cython.cdivision(True)
cdef void pot_nrg_hess_inter(
    int inter_flags                 , Py_ssize_t segm_size      , Py_ssize_t geodim ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    if inter_flags == 0:

        pot_nrg_hess_inter_size_law_nd(
            segm_size   , geodim    ,      
            pos_in      , dpos_in   , hess_in   ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 1:

        pot_nrg_hess_inter_store_law_nd(
            segm_size   , geodim    ,      
            pos_in      , dpos_in   , hess_in   ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 2:

        pot_nrg_hess_inter_size_law_2d(
            segm_size   ,      
            pos_in      , dpos_in   , hess_in   ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 3:

        pot_nrg_hess_inter_store_law_2d(
            segm_size   ,      
            pos_in      , dpos_in   , hess_in   ,
            inter_law   , inter_law_param_ptr   ,
        )

    elif inter_flags == 4:

        pot_nrg_hess_inter_size_gravity_nd(
            segm_size   , geodim    ,      
            pos_in      , dpos_in   , hess_in   ,
        )

    elif inter_flags == 5:

        pot_nrg_hess_inter_store_gravity_nd(
            segm_size   , geodim    ,      
            pos_in      , dpos_in   , hess_in   ,
        )

    elif inter_flags == 6:

        pot_nrg_hess_inter_size_gravity_2d(
            segm_size   ,      
            pos_in      , dpos_in   , hess_in   ,
        )

    elif inter_flags == 7:

        pot_nrg_hess_inter_store_gravity_2d(
            segm_size   ,      
            pos_in      , dpos_in   , hess_in   ,
        )

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_size_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    for iint in range(segm_size):

        dx2 = pos[0]*pos[0]
        dxtddx = pos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += pos[idim]*pos[idim]
            dxtddx += pos[idim]*dpos[idim]

        inter_law(dx2, pot, inter_law_param_ptr)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        for idim in range(geodim):
            hess[idim] = b*pos[idim]+a*dpos[idim]

        pos += geodim
        dpos += geodim
        hess += geodim

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_store_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    # First iteration
    dx2 = pos[0]*pos[0]
    dxtddx = pos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += pos[idim]*pos[idim]
        dxtddx += pos[idim]*dpos[idim]

    inter_law(dx2, pot, inter_law_param_ptr)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    for idim in range(geodim):
        hess[idim] = b*pos[idim]+a*dpos[idim]

    pos += geodim
    dpos += geodim
    hess += geodim

    for iint in range(1,segm_size):

        dx2 = pos[0]*pos[0]
        dxtddx = pos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += pos[idim]*pos[idim]
            dxtddx += pos[idim]*dpos[idim]

        inter_law(dx2, pot, inter_law_param_ptr)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        for idim in range(geodim):
            hess[idim] = b*pos[idim]+a*dpos[idim]

        pos += geodim
        dpos += geodim
        hess += geodim

    # Last iteration
    dx2 = pos[0]*pos[0]
    dxtddx = pos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += pos[idim]*pos[idim]
        dxtddx += pos[idim]*dpos[idim]

    inter_law(dx2, pot, inter_law_param_ptr)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    for idim in range(geodim):
        hess[idim] = b*pos[idim]+a*dpos[idim]

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_size_law_2d(
    Py_ssize_t segm_size            ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    for iint in range(segm_size):

        dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
        dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

        inter_law(dx2, pot, inter_law_param_ptr)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        hess[0] = b*pos[0]+a*dpos[0]
        hess[1] = b*pos[1]+a*dpos[1]

        pos += 2
        dpos += 2
        hess += 2

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_store_law_2d(
    Py_ssize_t segm_size            ,
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    # First iteration
    dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
    dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

    inter_law(dx2, pot, inter_law_param_ptr)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    hess[0] = b*pos[0]+a*dpos[0]
    hess[1] = b*pos[1]+a*dpos[1]

    pos += 2
    dpos += 2
    hess += 2

    for iint in range(1,segm_size):

        dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
        dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

        inter_law(dx2, pot, inter_law_param_ptr)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        hess[0] = b*pos[0]+a*dpos[0]
        hess[1] = b*pos[1]+a*dpos[1]

        pos += 2
        dpos += 2
        hess += 2

    # Last iteration
    dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
    dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

    inter_law(dx2, pot, inter_law_param_ptr)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    hess[0] = b*pos[0]+a*dpos[0]
    hess[1] = b*pos[1]+a*dpos[1]

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_size_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,      
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    for iint in range(segm_size):

        dx2 = pos[0]*pos[0]
        dxtddx = pos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += pos[idim]*pos[idim]
            dxtddx += pos[idim]*dpos[idim]

        inline_gravity_pot(dx2, pot)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        for idim in range(geodim):
            hess[idim] = b*pos[idim]+a*dpos[idim]

        pos += geodim
        dpos += geodim
        hess += geodim

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_store_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,      
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint, idim
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    # First iteration
    dx2 = pos[0]*pos[0]
    dxtddx = pos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += pos[idim]*pos[idim]
        dxtddx += pos[idim]*dpos[idim]

    inline_gravity_pot(dx2, pot)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    for idim in range(geodim):
        hess[idim] = b*pos[idim]+a*dpos[idim]

    pos += geodim
    dpos += geodim
    hess += geodim

    for iint in range(1,segm_size):

        dx2 = pos[0]*pos[0]
        dxtddx = pos[0]*dpos[0]
        for idim in range(1,geodim):
            dx2 += pos[idim]*pos[idim]
            dxtddx += pos[idim]*dpos[idim]

        inline_gravity_pot(dx2, pot)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        for idim in range(geodim):
            hess[idim] = b*pos[idim]+a*dpos[idim]

        pos += geodim
        dpos += geodim
        hess += geodim

    # Last iteration
    dx2 = pos[0]*pos[0]
    dxtddx = pos[0]*dpos[0]
    for idim in range(1,geodim):
        dx2 += pos[idim]*pos[idim]
        dxtddx += pos[idim]*dpos[idim]

    inline_gravity_pot(dx2, pot)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    for idim in range(geodim):
        hess[idim] = b*pos[idim]+a*dpos[idim]

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_size_gravity_2d(
    Py_ssize_t segm_size    ,      
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    for iint in range(segm_size):

        dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
        dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

        inline_gravity_pot(dx2, pot)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        hess[0] = b*pos[0]+a*dpos[0]
        hess[1] = b*pos[1]+a*dpos[1]

        pos += 2
        dpos += 2
        hess += 2

@cython.cdivision(True)
cdef void pot_nrg_hess_inter_store_gravity_2d(
    Py_ssize_t segm_size    ,
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil:

    cdef Py_ssize_t iint
    cdef double dx2, dxtddx, a, b
    cdef double[3] pot

    cdef double* pos = pos_in
    cdef double* dpos = dpos_in
    cdef double* hess = hess_in

    # First iteration
    dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
    dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

    inline_gravity_pot(dx2, pot)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    hess[0] = b*pos[0]+a*dpos[0]
    hess[1] = b*pos[1]+a*dpos[1]

    pos += 2
    dpos += 2
    hess += 2

    for iint in range(1,segm_size):

        dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
        dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

        inline_gravity_pot(dx2, pot)

        a = pot[1]
        b = 2*pot[2]*dxtddx

        hess[0] = b*pos[0]+a*dpos[0]
        hess[1] = b*pos[1]+a*dpos[1]

        pos += 2
        dpos += 2
        hess += 2

    # Last iteration
    dx2 = pos[0]*pos[0] + pos[1]*pos[1] 
    dxtddx = pos[0]*dpos[0] + pos[1]*dpos[1]

    inline_gravity_pot(dx2, pot)

    a = 0.5*pot[1]
    b = pot[2]*dxtddx

    hess[0] = b*pos[0]+a*dpos[0]
    hess[1] = b*pos[1]+a*dpos[1]

@cython.cdivision(True)
cdef void segmpos_to_unary_path_stats(
    double[:,:,::1] segmpos     ,
    double[:,:,::1] segmvel     ,
    Py_ssize_t segm_size        ,
    Py_ssize_t segm_store       ,
    double[::1]  out_segm_len   ,
) noexcept nogil:

    cdef int nsegm = segmpos.shape[0]
    cdef int geodim = segmpos.shape[2]

    cdef Py_ssize_t isegm, iint, jint
    cdef double vsq

    cdef double segm_len_val

    if (segm_size == segm_store):

        for isegm in range(nsegm):

            segm_len_val = 0

            for iint in range(segm_size):

                vsq = 0
                for idim in range(geodim):
                    vsq += segmvel[isegm,iint,idim] * segmvel[isegm,iint,idim]

                segm_len_val += csqrt(vsq)

            out_segm_len[isegm] = segm_len_val / segm_size

    else:

        for isegm in range(nsegm):

            vsq = 0
            for idim in range(geodim):
                vsq += segmvel[isegm,0,idim] * segmvel[isegm,0,idim]

            segm_len_val = csqrt(vsq) / 2

            for iint in range(1,segm_size):

                vsq = 0
                for idim in range(geodim):
                    vsq += segmvel[isegm,iint,idim] * segmvel[isegm,iint,idim]

                segm_len_val += csqrt(vsq)

            vsq = 0
            for idim in range(geodim):
                vsq += segmvel[isegm,segm_size,idim] * segmvel[isegm,segm_size,idim]

            segm_len_val += csqrt(vsq) / 2

            out_segm_len[isegm] = segm_len_val / segm_size

@cython.cdivision(True)
cdef void segmpos_to_binary_path_stats(
    double[:,:,::1] segmpos         ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    Py_ssize_t segm_store           ,
    double[::1]  out_bin_dx_min     ,
) noexcept nogil:

    cdef Py_ssize_t nbin = BinSourceSegm.shape[0]
    cdef int geodim = BinSpaceRot.shape[1]
    cdef int segm_store_int = segm_store
    cdef int nitems_int = segm_store*geodim
    cdef Py_ssize_t ibin, idim
    cdef Py_ssize_t isegm, isegmp
    cdef Py_ssize_t iint

    cdef double dx2
    cdef double dx2_min

    cdef double* tmp_loc_dpos = <double*> malloc(sizeof(double)*segm_store*geodim)
    cdef double* dpos

    for ibin in range(nbin):

        dpos = tmp_loc_dpos

        isegm = BinSourceSegm[ibin]
        isegmp = BinTargetSegm[ibin]

        if BinSpaceRotIsId[ibin]:
            scipy.linalg.cython_blas.dcopy(&nitems_int, &segmpos[isegm,0,0], &int_one, tmp_loc_dpos, &int_one)

        else:
            scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &BinSpaceRot[ibin,0,0], &geodim, &segmpos[isegm,0,0], &geodim, &zero_double, tmp_loc_dpos, &geodim)

        scipy.linalg.cython_blas.daxpy(&nitems_int, &minusone_double, &segmpos[isegmp,0,0], &int_one, tmp_loc_dpos, &int_one)

        dx2_min = DBL_MAX
        for iint in range(segm_store):

            dx2 = dpos[0]*dpos[0]
            for idim in range(1,geodim):
                dx2 += dpos[idim]*dpos[idim]

            dx2_min = min(dx2_min, dx2)
            dpos += geodim

        out_bin_dx_min[ibin] = csqrt(dx2_min)

    free(tmp_loc_dpos)

cdef inline void Compute_forces_vectorized(
    double* pos                     , double* forces                    ,
    Py_ssize_t nbin                 , Py_ssize_t geodim                 ,
    Py_ssize_t nsegm                , Py_ssize_t nvec                   ,
    Py_ssize_t* BinSourceSegm       , Py_ssize_t* BinTargetSegm         ,
    double* BinSpaceRot             , bint* BinSpaceRotIsId             ,
    double* BinProdChargeSumSource  , double* BinProdChargeSumTarget    ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr         ,
) noexcept nogil:

    cdef Py_ssize_t ibin, idim, jdim, ivec
    cdef Py_ssize_t isegm, isegmp
    cdef Py_ssize_t geodim_sq = geodim * geodim
    cdef Py_ssize_t vec_size = nsegm * geodim

    cdef double* matel
    cdef double* vec_pos
    cdef double* vec_forces
    cdef double* cur_pos
    cdef double* cur_posp
    cdef double* cur_forces

    cdef double dx2
    cdef double bin_fac
    cdef double[3] pot

    cdef double* dx = <double*> malloc(sizeof(double)*geodim)
    cdef double* df = <double*> malloc(sizeof(double)*geodim)

    cdef Py_ssize_t nmem = nvec * nsegm * geodim
    memset(forces, 0, sizeof(double)*nmem)

    for ivec in range(nvec):

        vec_pos     = pos    + ivec * vec_size
        vec_forces  = forces + ivec * vec_size

        for ibin in range(nbin):

            isegm = BinSourceSegm[ibin]
            isegmp = BinTargetSegm[ibin]

            if BinSpaceRotIsId[ibin]:

                cur_pos  = vec_pos + isegm *geodim
                cur_posp = vec_pos + isegmp*geodim
                    
                for idim in range(geodim):
                    dx[idim] = cur_pos[0] - cur_posp[0]
                    cur_pos  += 1
                    cur_posp += 1

            else:

                matel = BinSpaceRot + ibin*geodim_sq
                cur_posp = vec_pos + isegmp*geodim

                for idim in range(geodim):
                    dx[idim] = -cur_posp[0]
                    cur_pos  = vec_pos + isegm *geodim
                    for jdim in range(geodim):
                        dx[idim] += matel[0] * cur_pos[0]
                        matel += 1 
                        cur_pos += 1

                    cur_posp += 1

            dx2 = dx[0]*dx[0]
            for idim in range(1,geodim):
                dx2 += dx[idim]*dx[idim]

            inter_law(dx2, pot, inter_law_param_ptr)

            if BinProdChargeSumSource[ibin] != 0.:

                bin_fac = pot[1]*(-4)*BinProdChargeSumSource[ibin]
            
                for idim in range(geodim):
                    df[idim] = bin_fac*dx[idim]

                if BinSpaceRotIsId[ibin]:

                    cur_forces = vec_forces + isegm*geodim

                    for idim in range(geodim):
                        cur_forces[0] += df[idim]
                        cur_forces += 1
                        
                else:

                    matel = BinSpaceRot + ibin*geodim_sq

                    for jdim in range(geodim):

                        cur_forces = vec_forces + isegm*geodim

                        for idim in range(geodim):
                            cur_forces[0] += matel[0] * df[jdim]
                            matel += 1 
                            cur_forces += 1

            if BinProdChargeSumTarget[ibin] != 0.:

                bin_fac = pot[1]*4*BinProdChargeSumTarget[ibin]

                for idim in range(geodim):
                    df[idim] = bin_fac*dx[idim]

                cur_forces = vec_forces + isegmp*geodim

                for idim in range(geodim):
                    cur_forces[0] += df[idim]
                    cur_forces += 1
                
    free(dx)
    free(df)

cdef inline void Compute_grad_forces_vectorized(
    double* pos                     , double* dpos                      , double* grad_forces   ,
    Py_ssize_t nbin                 , Py_ssize_t geodim                 ,
    Py_ssize_t nsegm                , Py_ssize_t nvec                   , Py_ssize_t grad_ndof  ,
    Py_ssize_t* BinSourceSegm       , Py_ssize_t* BinTargetSegm         ,
    double* BinSpaceRot             , bint* BinSpaceRotIsId             ,
    double* BinProdChargeSumSource  , double* BinProdChargeSumTarget    ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr         ,
) noexcept nogil:

    cdef Py_ssize_t ibin, idim, jdim, ivec
    cdef Py_ssize_t isegm, isegmp
    cdef Py_ssize_t geodim_sq = geodim * geodim
    cdef Py_ssize_t vec_size = nsegm * geodim
    cdef Py_ssize_t vec_grad_size = vec_size * grad_ndof
    cdef Py_ssize_t i_grad_dof

    cdef int geodim_int = geodim
    cdef int grad_ndof_int = grad_ndof
    cdef int dsegm_size = geodim * grad_ndof

    cdef double* RotMat
    cdef double* vec_pos
    cdef double* vec_dpos
    cdef double* vec_grad_forces
    cdef double* grad_forces_loc
    cdef double* grad_forces_locp

    cdef double* cur_pos
    cdef double* cur_posp
    cdef double* cur_dpos
    cdef double* cur_dposp

    cdef double dx2, dxtddx, a, b
    cdef double bin_fac
    cdef double[3] pot

    cdef double* dx = <double*> malloc(sizeof(double)*geodim)
    cdef double* ddx = <double*> malloc(sizeof(double)*dsegm_size)
    cdef double* ddf = <double*> malloc(sizeof(double)*dsegm_size)

    cdef double* ddx_cur
    cdef double* ddf_cur

    cdef Py_ssize_t nmem = nvec * vec_grad_size
    memset(grad_forces, 0, sizeof(double)*nmem)

    for ivec in range(nvec):

        vec_pos  = pos   + ivec * vec_size
        vec_dpos = dpos  + ivec * vec_grad_size
        vec_grad_forces = grad_forces + ivec * vec_grad_size

        for ibin in range(nbin):

            isegm = BinSourceSegm[ibin]
            isegmp = BinTargetSegm[ibin]

            if BinSpaceRotIsId[ibin]:

                cur_pos  = vec_pos  + isegm * geodim
                cur_posp = vec_pos  + isegmp * geodim
                    
                for idim in range(geodim):
                    dx[idim] = cur_pos[0] - cur_posp[0]

                    cur_pos  += 1
                    cur_posp += 1

            else:

                RotMat = BinSpaceRot + ibin * geodim_sq
                cur_posp  = vec_pos  + isegmp * geodim

                for idim in range(geodim):
                    
                    dx[idim]  = -cur_posp[0]
                    cur_pos   = vec_pos + isegm * geodim

                    for jdim in range(geodim):
                        dx[idim] += RotMat[0] * cur_pos[0]
                    
                        RotMat += 1 
                        cur_pos += 1

                    cur_posp += 1

            dx2 = dx[0]*dx[0]
            for idim in range(1,geodim):
                dx2 += dx[idim]*dx[idim]

            inter_law(dx2, pot, inter_law_param_ptr)

            a = pot[1]
            pot[2] *= 2

            cur_dpos  = vec_dpos + isegm  * geodim * grad_ndof
            cur_dposp = vec_dpos + isegmp * geodim * grad_ndof

            if BinSpaceRotIsId[ibin]:
                scipy.linalg.cython_blas.dcopy(&dsegm_size,cur_dpos,&int_one,ddx,&int_one)

            else:

                RotMat = BinSpaceRot + ibin * geodim_sq
                scipy.linalg.cython_blas.dgemm(transn, transn, &grad_ndof_int, &geodim_int, &geodim_int, &one_double, cur_dpos, &grad_ndof_int, RotMat, &geodim_int, &zero_double, ddx, &grad_ndof_int)

            scipy.linalg.cython_blas.daxpy(&dsegm_size,&minusone_double, cur_dposp, &int_one, ddx, &int_one)

            ddx_cur = ddx
            ddf_cur = ddf

            # TODO: Remove this for loop ?
            for i_grad_dof in range(grad_ndof):

                dxtddx = dx[0]*ddx_cur[0]
                for idim in range(1,geodim):
                    dxtddx += dx[idim]*ddx_cur[idim*grad_ndof]

                b = pot[2]*dxtddx

                for idim in range(geodim):
                    ddf_cur[idim*grad_ndof] = b*dx[idim]+a*ddx_cur[idim*grad_ndof]

                ddx_cur += 1
                ddf_cur += 1

            grad_forces_loc  = vec_grad_forces + isegm  * geodim * grad_ndof
            grad_forces_locp = vec_grad_forces + isegmp * geodim * grad_ndof

            if BinProdChargeSumSource[ibin] != 0.:

                bin_fac = (-4)*BinProdChargeSumSource[ibin]

                if BinSpaceRotIsId[ibin]:
                    scipy.linalg.cython_blas.daxpy(&dsegm_size,&bin_fac, ddf, &int_one, grad_forces_loc, &int_one)

                else:

                    RotMat = BinSpaceRot + ibin * geodim_sq
                    scipy.linalg.cython_blas.dgemm(transn, transt, &grad_ndof_int, &geodim_int, &geodim_int, &bin_fac, ddf, &grad_ndof_int, RotMat, &geodim_int, &one_double, grad_forces_loc, &grad_ndof_int)

            if BinProdChargeSumTarget[ibin] != 0.:

                bin_fac = 4*BinProdChargeSumTarget[ibin]
                scipy.linalg.cython_blas.daxpy(&dsegm_size,&bin_fac, ddf, &int_one, grad_forces_locp, &int_one)
                    
    free(dx)
    free(ddx)
    free(ddf)

cdef void Compute_forces_user_data(
    double t    , double[::1] pos   , double[::1] forces    , void* user_data   ,
) noexcept nogil:

    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_forces_vectorized(
        &pos[0]                                 , &forces[0]                            ,
        ODE_params.nbin                         , ODE_params.geodim                     , 
        ODE_params.nsegm                        , 1                                     ,
        ODE_params.BinSourceSegm_ptr            , ODE_params.BinTargetSegm_ptr          ,
        ODE_params.BinSpaceRot_ptr              , ODE_params.BinSpaceRotIsId_ptr        ,
        ODE_params.BinProdChargeSumSource_ptr   , ODE_params.BinProdChargeSumTarget_ptr ,
        ODE_params.inter_law                    , ODE_params.inter_law_param_ptr        ,
    )

cdef void Compute_grad_forces_user_data(
    double t    , double[::1] pos   , double[:,::1] dpos   , double[:,::1] grad_forces  , void* user_data   ,
) noexcept nogil:

    cdef Py_ssize_t grad_ndof = dpos.shape[1]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_grad_forces_vectorized(
        &pos[0]                                 , &dpos[0,0]                            , &grad_forces[0,0] ,
        ODE_params.nbin                         , ODE_params.geodim                     , 
        ODE_params.nsegm                        , 1                                     , grad_ndof         ,
        ODE_params.BinSourceSegm_ptr            , ODE_params.BinTargetSegm_ptr          ,
        ODE_params.BinSpaceRot_ptr              , ODE_params.BinSpaceRotIsId_ptr        ,
        ODE_params.BinProdChargeSumSource_ptr   , ODE_params.BinProdChargeSumTarget_ptr ,
        ODE_params.inter_law                    , ODE_params.inter_law_param_ptr        ,
    )

cdef void Compute_forces_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,::1] all_forces, void* user_data ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_pos.shape[0]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_forces_vectorized(
        &all_pos[0,0]                           , &all_forces[0,0]                      ,
        ODE_params.nbin                         , ODE_params.geodim                     ,
        ODE_params.nsegm                        , nvec                                  ,
        ODE_params.BinSourceSegm_ptr            , ODE_params.BinTargetSegm_ptr          ,
        ODE_params.BinSpaceRot_ptr              , ODE_params.BinSpaceRotIsId_ptr        ,
        ODE_params.BinProdChargeSumSource_ptr   , ODE_params.BinProdChargeSumTarget_ptr ,
        ODE_params.inter_law                    , ODE_params.inter_law_param_ptr        ,
    )

cdef void Compute_grad_forces_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,:,::1] all_dpos  , double[:,:,::1] all_grad_forces   , void* user_data ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_pos.shape[0]
    cdef Py_ssize_t grad_ndof = all_dpos.shape[2]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_grad_forces_vectorized(
        &all_pos[0,0]                           , &all_dpos[0,0,0]                      , &all_grad_forces[0,0,0]   ,
        ODE_params.nbin                         , ODE_params.geodim                     ,
        ODE_params.nsegm                        , nvec                                  , grad_ndof                 ,
        ODE_params.BinSourceSegm_ptr            , ODE_params.BinTargetSegm_ptr          ,
        ODE_params.BinSpaceRot_ptr              , ODE_params.BinSpaceRotIsId_ptr        ,
        ODE_params.BinProdChargeSumSource_ptr   , ODE_params.BinProdChargeSumTarget_ptr ,
        ODE_params.inter_law                    , ODE_params.inter_law_param_ptr        ,
    )

cdef inline void Compute_forces_vectorized_nosym(
    double* pos                     , double* forces            ,
    Py_ssize_t geodim               ,
    Py_ssize_t nsegm                , Py_ssize_t nvec           ,
    double* SegmCharge              ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t idim, jdim, ivec
    cdef Py_ssize_t isegm, isegmp
    cdef Py_ssize_t vec_size = nsegm * geodim

    cdef double* vec_pos
    cdef double* vec_forces
    cdef double* cur_pos
    cdef double* cur_posp
    cdef double* cur_forces

    cdef double dx2
    cdef double bin_fac
    cdef double[3] pot

    cdef double* dx = <double*> malloc(sizeof(double)*geodim)
    cdef double* df = <double*> malloc(sizeof(double)*geodim)

    cdef Py_ssize_t nmem = nvec * nsegm * geodim
    memset(forces, 0, sizeof(double)*nmem)

    for ivec in range(nvec):

        vec_pos     = pos    + ivec * vec_size
        vec_forces  = forces + ivec * vec_size

        for isegm in range(nsegm-1):
            for isegmp in range(isegm+1, nsegm):

                cur_pos  = vec_pos + isegm *geodim
                cur_posp = vec_pos + isegmp*geodim
                    
                for idim in range(geodim):
                    dx[idim] = cur_pos[0] - cur_posp[0]
                    cur_pos  += 1
                    cur_posp += 1

                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]

                inter_law(dx2, pot, inter_law_param_ptr)

                bin_fac = (-4)*SegmCharge[isegm]*SegmCharge[isegmp]

                pot[1] *= bin_fac

                for idim in range(geodim):
                    df[idim] = pot[1]*dx[idim]

                cur_forces = vec_forces + isegm*geodim

                for idim in range(geodim):
                    cur_forces[0] += df[idim]
                    cur_forces += 1
                        
                cur_forces = vec_forces + isegmp*geodim

                for idim in range(geodim):
                    cur_forces[0] -= df[idim]
                    cur_forces += 1
                    
    free(dx)
    free(df)

cdef inline void Compute_grad_forces_vectorized_nosym(
    double* pos                     , double* dpos              , double* grad_forces   ,
    Py_ssize_t geodim               ,
    Py_ssize_t nsegm                , Py_ssize_t nvec           , Py_ssize_t grad_ndof  ,
    double* SegmCharge              ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil:

    cdef Py_ssize_t idim, jdim, ivec
    cdef Py_ssize_t isegm, isegmp
    cdef Py_ssize_t vec_size = nsegm * geodim

    cdef int geodim_int = geodim
    cdef int grad_ndof_int = grad_ndof
    cdef int dsegm_size = geodim * grad_ndof

    cdef double* vec_pos
    cdef double* vec_dpos
    cdef double* cur_pos
    cdef double* cur_posp
    cdef double* cur_dpos
    cdef double* cur_dposp
    cdef double* vec_grad_forces
    cdef double* grad_forces_loc
    cdef double* grad_forces_locp

    cdef double dx2, dxtddx, a ,b
    cdef double bin_fac
    cdef double[3] pot

    cdef double* dx = <double*> malloc(sizeof(double)*geodim)
    cdef double* ddx = <double*> malloc(sizeof(double)*geodim*grad_ndof)
    cdef double* ddf = <double*> malloc(sizeof(double)*geodim*grad_ndof)

    cdef double* ddx_cur
    cdef double* ddf_cur

    cdef Py_ssize_t nmem = nvec * vec_size * grad_ndof
    memset(grad_forces, 0, sizeof(double)*nmem)

    for ivec in range(nvec):

        vec_pos         = pos         + ivec * vec_size
        vec_dpos        = dpos        + ivec * vec_size * grad_ndof
        vec_grad_forces = grad_forces + ivec * vec_size * grad_ndof

        for isegm in range(nsegm-1):

            cur_dpos  = vec_dpos + isegm  * geodim * grad_ndof
            grad_forces_loc  = vec_grad_forces + isegm  * geodim * grad_ndof
        
            for isegmp in range(isegm+1, nsegm):

                cur_pos  = vec_pos + isegm *geodim
                cur_posp = vec_pos + isegmp*geodim
                    
                for idim in range(geodim):
                    dx[idim] = cur_pos[0] - cur_posp[0]
                    cur_pos  += 1
                    cur_posp += 1

                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]

                inter_law(dx2, pot, inter_law_param_ptr)

                bin_fac = (-4)*SegmCharge[isegm]*SegmCharge[isegmp]
                
                a = pot[1]*bin_fac
                pot[2] *= 2*bin_fac

                cur_dposp = vec_dpos + isegmp * geodim * grad_ndof

                scipy.linalg.cython_blas.dcopy(&dsegm_size,cur_dpos,&int_one,ddx,&int_one)
                scipy.linalg.cython_blas.daxpy(&dsegm_size,&minusone_double, cur_dposp, &int_one, ddx, &int_one)

                ddx_cur = ddx
                ddf_cur = ddf

                # TODO: Remove this for loop ?
                for i_grad_dof in range(grad_ndof):

                    dxtddx = dx[0]*ddx_cur[0]
                    for idim in range(1,geodim):
                        dxtddx += dx[idim]*ddx_cur[idim*grad_ndof]

                    b = pot[2]*dxtddx

                    for idim in range(geodim):
                        ddf_cur[idim*grad_ndof] = b*dx[idim]+a*ddx_cur[idim*grad_ndof]

                    ddx_cur += 1
                    ddf_cur += 1

                grad_forces_locp = vec_grad_forces + isegmp * geodim * grad_ndof

                scipy.linalg.cython_blas.daxpy(&dsegm_size,&one_double, ddf, &int_one, grad_forces_loc, &int_one)
                scipy.linalg.cython_blas.daxpy(&dsegm_size,&minusone_double, ddf, &int_one, grad_forces_locp, &int_one)
                    
    free(dx)
    free(ddx)
    free(ddf)

cdef void Compute_forces_nosym_user_data(
    double t    , double[::1] pos   , double[::1] forces    , void* user_data   ,
) noexcept nogil:

    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_forces_vectorized_nosym(
        &pos[0]                         , &forces[0]                        ,
        ODE_params.geodim               , 
        ODE_params.nsegm                , 1                                 ,
        ODE_params.SegmCharge           ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
    )

cdef void Compute_grad_forces_nosym_user_data(
    double t    , double[::1] pos   , double[:,::1] dpos    , double[:,::1] grad_forces , void* user_data   ,
) noexcept nogil:

    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    cdef Py_ssize_t grad_ndof = dpos.shape[1]

    Compute_grad_forces_vectorized_nosym(
        &pos[0]                         , &dpos[0,0]                        , &grad_forces[0,0] ,
        ODE_params.geodim               , 
        ODE_params.nsegm                , 1                                 , grad_ndof         ,
        ODE_params.SegmCharge           ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
    )

cdef void Compute_forces_vectorized_nosym_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,::1] all_forces, void* user_data ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_pos.shape[0]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_forces_vectorized_nosym(
        &all_pos[0,0]                   , &all_forces[0,0]                  ,
        ODE_params.geodim               , 
        ODE_params.nsegm                , nvec                              ,
        ODE_params.SegmCharge           ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
    )

cdef void Compute_grad_forces_vectorized_nosym_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,:,::1] all_dpos  , double[:,:,::1] all_grad_forces   , void* user_data ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_pos.shape[0]
    cdef Py_ssize_t grad_ndof = all_dpos.shape[2]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_grad_forces_vectorized_nosym(
        &all_pos[0,0]                   , &all_dpos[0,0,0]                  , &all_grad_forces[0,0,0]   ,
        ODE_params.geodim               , 
        ODE_params.nsegm                , nvec                              , grad_ndof                 ,
        ODE_params.SegmCharge           ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
    )

cdef inline void Compute_velocities_vectorized(
    double* mom         , double* res       ,
    Py_ssize_t nbin     , Py_ssize_t geodim ,
    Py_ssize_t nsegm    , Py_ssize_t nvec   ,
    double* InvSegmMass , 
) noexcept nogil:

    cdef Py_ssize_t isegm, idim, ivec
    cdef double* cur_mom
    cdef double* cur_res

    cdef double invmass 

    cur_mom = mom
    cur_res = res

    for ivec in range(nvec):

        for isegm in range(nsegm):

            invmass = InvSegmMass[isegm]

            for idim in range(geodim):

                cur_res[0] = cur_mom[0] * invmass

                cur_mom += 1
                cur_res += 1

cdef inline void Compute_grad_velocities_vectorized(
    double* mom         , double* grad_mom  , double* res           ,
    Py_ssize_t nbin     , Py_ssize_t geodim ,
    Py_ssize_t nsegm    , Py_ssize_t nvec   , Py_ssize_t grad_ndof  ,
    double* InvSegmMass , 
) noexcept nogil:

    cdef Py_ssize_t isegm, idim, ivec
    cdef int segm_grad_size = geodim * grad_ndof
    cdef int ndof_tot = segm_grad_size * nvec * nsegm
    cdef double* cur_res

    scipy.linalg.cython_blas.dcopy(&ndof_tot,grad_mom,&int_one,res,&int_one)

    cur_res = res

    for ivec in range(nvec):

        for isegm in range(nsegm):

            scipy.linalg.cython_blas.dscal(&segm_grad_size,&InvSegmMass[isegm],cur_res,&int_one)

            cur_res += segm_grad_size

cdef void Compute_velocities_user_data(
    double t    , double[::1] mom   , double[::1] res   , void* user_data   ,
) noexcept nogil:

    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_velocities_vectorized(
        &mom[0]                 , &res[0]           ,
        ODE_params.nbin         , ODE_params.geodim , 
        ODE_params.nsegm        , 1                 ,
        ODE_params.InvSegmMass  ,
    )

cdef void Compute_grad_velocities_user_data(
    double t    , double[::1] mom   , double[:,::1] grad_mom  , double[:,::1] res   , void* user_data   ,
) noexcept nogil:

    cdef Py_ssize_t grad_ndof = grad_mom.shape[1]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_grad_velocities_vectorized(
        &mom[0]                 , &grad_mom[0,0]    , &res[0,0] ,
        ODE_params.nbin         , ODE_params.geodim , 
        ODE_params.nsegm        , 1                 , grad_ndof , 
        ODE_params.InvSegmMass  ,
    )

cdef void Compute_velocities_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_mom , double[:,::1] all_res , void* user_data   ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_res.shape[0]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_velocities_vectorized(
        &all_mom[0,0]           , &all_res[0,0]     ,
        ODE_params.nbin         , ODE_params.geodim , 
        ODE_params.nsegm        , nvec              ,
        ODE_params.InvSegmMass  ,
    )

cdef void Compute_grad_velocities_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_mom , double[:,:,::1] all_grad_mom , double[:,:,::1] all_res , void* user_data   ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_res.shape[0]
    cdef Py_ssize_t grad_ndof = all_grad_mom.shape[2]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_grad_velocities_vectorized(
        &all_mom[0,0]           , &all_grad_mom[0,0,0]  , &all_res[0,0,0]   ,
        ODE_params.nbin         , ODE_params.geodim     , 
        ODE_params.nsegm        , nvec                  , grad_ndof         , 
        ODE_params.InvSegmMass  ,
    )
