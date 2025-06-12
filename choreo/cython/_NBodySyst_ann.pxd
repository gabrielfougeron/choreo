cimport numpy as np
cimport cython
from choreo.cython.optional_pyfftw cimport pyfftw

cdef int USE_SCIPY_FFT
cdef int USE_MKL_FFT
cdef int USE_FFTW_FFT
cdef int USE_DUCC_FFT

cdef int GENERAL_SYM
cdef int RFFT

ctypedef void (*inter_law_fun_type)(double, double*, void*) noexcept nogil 

cdef void inline_gravity_pot(double xsq, double* res) noexcept nogil
cdef void gravity_pot(double xsq, double* res, void* pot_params) noexcept nogil
cdef void power_law_pot(double xsq, double* res, void* pot_params) noexcept nogil

cdef double[::1] default_Hash_exp

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

cdef void Make_Init_bounds_coeffs(
    double *params_pos_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    double coeff_ampl_o             , double coeff_ampl_min             ,
    Py_ssize_t k_infl               , Py_ssize_t k_max                  ,
) noexcept nogil

cdef void changevar_mom_pos(
    double *params_mom_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double **params_pos_buf         , 
) noexcept nogil

cdef void changevar_mom_pos_invT(
    double *params_mom_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double **params_pos_buf         , 
) noexcept nogil

cdef void changevar_mom_pos_inv(
    double **params_pos_buf         , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double *params_mom_buf          , 
) noexcept nogil

cdef void changevar_mom_pos_T(
    double **params_pos_buf         , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double *params_mom_buf          ,   
) noexcept nogil

cdef void changevar_mom_vel(
    double *params_mom_buf          , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] nnz_k_buf       , Py_ssize_t[:,::1] nnz_k_shapes    , Py_ssize_t[::1] nnz_k_shifts  ,
    bint[::1] co_in_buf             , Py_ssize_t[:,::1] co_in_shapes    , Py_ssize_t[::1] co_in_shifts  ,
    Py_ssize_t[::1] ncoeff_min_loop ,
    Py_ssize_t[::1] loopnb          , double[::1] loopmass              ,
    double **params_vel_buf         , 
) noexcept nogil

cdef double params_to_kin_nrg(
    double *params_mom_buf      , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] ncor_loop   , Py_ssize_t[::1] nco_in_loop       ,
) noexcept nogil

cdef void params_to_kin_nrg_grad_daxpy(
    double *params_mom_buf      , Py_ssize_t[:,::1] params_shapes   , Py_ssize_t[::1] params_shifts ,
    Py_ssize_t[::1] ncor_loop   , Py_ssize_t[::1] nco_in_loop       ,
    double mul                  ,
    double *grad_buf            ,
) noexcept nogil

cdef void inplace_twiddle(
    double complex* const_ifft  ,
    Py_ssize_t* nnz_k           ,
    Py_ssize_t nint             ,
    int n_inter                 ,
    int ncoeff_min_loop_nnz     ,
    int nppl                    ,
    int direction               , # -1 or 1
) noexcept nogil

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
) noexcept nogil

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
) noexcept nogil
 
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
) noexcept nogil

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
) noexcept nogil

cdef void Adjust_after_last_gen(
    double** pos_slice_buf_ptr              , Py_ssize_t[::1] pos_slice_shifts  ,
    Py_ssize_t[:,::1] ifft_shapes           ,
    Py_ssize_t[:,::1] params_basis_shapes   ,
    Py_ssize_t[::1] n_sub_fft               ,
    Py_ssize_t[::1] ALG_Iint                ,
    Py_ssize_t[::1] ALG_TimeRev             , double[:,:,::1] ALG_SpaceRot      ,
    Py_ssize_t segm_size                    ,
)noexcept nogil

cdef void Adjust_after_last_gen_T(
    double** pos_slice_buf_ptr              , Py_ssize_t[::1] pos_slice_shifts  ,
    Py_ssize_t[:,::1] ifft_shapes           ,
    Py_ssize_t[:,::1] params_basis_shapes   ,
    Py_ssize_t[::1] n_sub_fft               ,
    Py_ssize_t[::1] ALG_Iint                ,
    Py_ssize_t[::1] ALG_TimeRev             , double[:,:,::1] ALG_SpaceRot      ,
    Py_ssize_t segm_size                    ,
)noexcept nogil

cdef void params_to_pos_slice(
    double** params_buf                     , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    Py_ssize_t* nnz_k_buf_ptr               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    double complex **ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    int[::1] ParamBasisShortcut             ,
    int fft_backend                         , pyfftw.fftw_exe** fftw_genrfft_exe    , pyfftw.fftw_exe** fftw_symirfft_exe   ,
    double complex *params_basis_buf_ptr    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    double** pos_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             ,
) noexcept nogil

cdef void pos_slice_to_params(
    double** pos_slice_buf_ptr              , Py_ssize_t[:,::1] pos_slice_shapes    , Py_ssize_t[::1] pos_slice_shifts      ,
    double complex *params_basis_buf_ptr    , Py_ssize_t[:,::1] params_basis_shapes , Py_ssize_t[::1] params_basis_shifts   ,
    Py_ssize_t* nnz_k_buf_ptr               , Py_ssize_t[:,::1] nnz_k_shapes        , Py_ssize_t[::1] nnz_k_shifts          ,
    double complex **ifft_buf_ptr           , Py_ssize_t[:,::1] ifft_shapes         , Py_ssize_t[::1] ifft_shifts           ,
    Py_ssize_t[::1] ncoeff_min_loop         , Py_ssize_t[::1] n_sub_fft             , int direction                         ,
    double **params_buf                     , Py_ssize_t[:,::1] params_shapes       , Py_ssize_t[::1] params_shifts         ,
    int[::1] ParamBasisShortcut             ,
    int fft_backend                         , pyfftw.fftw_exe** fftw_genirfft_exe   , pyfftw.fftw_exe** fftw_symrfft_exe    ,
) noexcept nogil

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
) noexcept nogil

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
) noexcept nogil

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
) noexcept nogil

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
) noexcept nogil

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
) noexcept nogil

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
) noexcept nogil

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
) noexcept nogil

cdef int get_inter_flags(
    Py_ssize_t segm_size            , Py_ssize_t segm_store ,
    Py_ssize_t geodim               ,
    inter_law_fun_type inter_law
) noexcept nogil

cdef void segm_pos_to_hash(
    double[:,:,::1] segmpos         ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         ,
    double[::1] Hash_exp            , double[::1] Hash              ,           
) noexcept nogil

cdef double segm_pos_to_pot_nrg(
    double[:,:,::1] segmpos         ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr     ,
) noexcept nogil

cdef void segm_pos_to_pot_nrg_grad(
    double[:,:,::1] segmpos         , double[:,:,::1] pot_nrg_grad  ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         , double globalmul  ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr     ,
) noexcept nogil

cdef void pot_nrg_grad_inter(
    int inter_flags , Py_ssize_t segm_size  , Py_ssize_t geodim         ,      
    double* dpos_in , double* grad_in       ,
    inter_law_fun_type inter_law            , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_grad_inter_size_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_grad_inter_store_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_grad_inter_size_law_2d(
    Py_ssize_t segm_size            ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_grad_inter_store_law_2d(
    Py_ssize_t segm_size            ,      
    double* dpos_in                 , double* grad_in           ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_grad_inter_size_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,        
    double* dpos_in         , double* grad_in   ,
) noexcept nogil

cdef void pot_nrg_grad_inter_store_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,      
    double* dpos_in         , double* grad_in   ,
) noexcept nogil

cdef void pot_nrg_grad_inter_size_gravity_2d(
    Py_ssize_t segm_size    ,      
    double* dpos_in         , double* grad_in   ,
) noexcept nogil

cdef void pot_nrg_grad_inter_store_gravity_2d(
    Py_ssize_t segm_size    ,      
    double* dpos_in         , double* grad_in   ,
) noexcept nogil

cdef void segm_pos_to_pot_nrg_hess(
    double[:,:,::1] segmpos         , double[:,:,::1] dsegmpos      , double[:,:,::1] pot_nrg_hess  ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    double[::1] BinProdChargeSum    ,
    Py_ssize_t segm_size            , Py_ssize_t segm_store         , double globalmul              ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr     ,
) noexcept nogil

cdef void pot_nrg_hess_inter(
    int inter_flags                 , Py_ssize_t segm_size      , Py_ssize_t geodim ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_hess_inter_size_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_hess_inter_store_law_nd(
    Py_ssize_t segm_size            , Py_ssize_t geodim         ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_hess_inter_size_law_2d(
    Py_ssize_t segm_size            ,      
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_hess_inter_store_law_2d(
    Py_ssize_t segm_size            ,
    double* pos_in                  , double* dpos_in           , double* hess_in   ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void pot_nrg_hess_inter_size_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,      
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil

cdef void pot_nrg_hess_inter_store_gravity_nd(
    Py_ssize_t segm_size    , Py_ssize_t geodim ,      
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil

cdef void pot_nrg_hess_inter_size_gravity_2d(
    Py_ssize_t segm_size    ,      
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil

cdef void pot_nrg_hess_inter_store_gravity_2d(
    Py_ssize_t segm_size    ,
    double* pos_in          , double* dpos_in   , double* hess_in   ,
) noexcept nogil

cdef void segmpos_to_unary_path_stats(
    double[:,:,::1] segmpos     ,
    double[:,:,::1] segmvel     ,
    Py_ssize_t segm_size        ,
    Py_ssize_t segm_store       ,
    double[::1]  out_segm_len   ,
) noexcept nogil

cdef void segmpos_to_binary_path_stats(
    double[:,:,::1] segmpos         ,
    Py_ssize_t[::1] BinSourceSegm   , Py_ssize_t[::1] BinTargetSegm ,
    double[:,:,::1] BinSpaceRot     , bint[::1] BinSpaceRotIsId     ,
    Py_ssize_t segm_store           ,
    double[::1]  out_bin_dx_min     ,
) noexcept nogil

cdef void Compute_forces_vectorized(
    double* pos                     , double* forces                    ,
    Py_ssize_t nbin                 , Py_ssize_t geodim                 ,
    Py_ssize_t nsegm                , Py_ssize_t nvec                   ,
    Py_ssize_t* BinSourceSegm       , Py_ssize_t* BinTargetSegm         ,
    double* BinSpaceRot             , bint* BinSpaceRotIsId             ,
    double* BinProdChargeSumSource  , double* BinProdChargeSumTarget    ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr         ,
) noexcept nogil

cdef void Compute_grad_forces_vectorized(
    double* pos                     , double* dpos                      , double* grad_forces   ,
    Py_ssize_t nbin                 , Py_ssize_t geodim                 ,
    Py_ssize_t nsegm                , Py_ssize_t nvec                   , Py_ssize_t grad_ndof  ,
    Py_ssize_t* BinSourceSegm       , Py_ssize_t* BinTargetSegm         ,
    double* BinSpaceRot             , bint* BinSpaceRotIsId             ,
    double* BinProdChargeSumSource  , double* BinProdChargeSumTarget    ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr         ,
) noexcept nogil

cdef void Compute_forces_user_data(
    double t    , double[::1] pos   , double[::1] forces    , void* user_data   ,
) noexcept nogil

cdef void Compute_grad_forces_user_data(
    double t    , double[::1] pos   , double[:,::1] dpos   , double[:,::1] grad_forces  , void* user_data   ,
) noexcept nogil

cdef void Compute_forces_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,::1] all_forces, void* user_data ,
) noexcept nogil

cdef void Compute_grad_forces_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,:,::1] all_dpos  , double[:,:,::1] all_grad_forces   , void* user_data ,
) noexcept nogil

cdef void Compute_forces_vectorized_nosym(
    double* pos                     , double* forces            ,
    Py_ssize_t geodim               ,
    Py_ssize_t nsegm                , Py_ssize_t nvec           ,
    double* SegmCharge              ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void Compute_grad_forces_vectorized_nosym(
    double* pos                     , double* dpos              , double* grad_forces   ,
    Py_ssize_t geodim               ,
    Py_ssize_t nsegm                , Py_ssize_t nvec           , Py_ssize_t grad_ndof  ,
    double* SegmCharge              ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
) noexcept nogil

cdef void Compute_forces_nosym_user_data(
    double t    , double[::1] pos   , double[::1] forces    , void* user_data   ,
) noexcept nogil

cdef void Compute_grad_forces_nosym_user_data(
    double t    , double[::1] pos   , double[:,::1] dpos    , double[:,::1] grad_forces , void* user_data   ,
) noexcept nogil

cdef void Compute_forces_vectorized_nosym_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,::1] all_forces, void* user_data ,
) noexcept nogil

cdef void Compute_grad_forces_vectorized_nosym_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,:,::1] all_dpos  , double[:,:,::1] all_grad_forces   , void* user_data ,
) noexcept nogil

cdef void Compute_velocities_vectorized(
    double* mom         , double* res       ,
    Py_ssize_t nbin     , Py_ssize_t geodim ,
    Py_ssize_t nsegm    , Py_ssize_t nvec   ,
    double* InvSegmMass , 
) noexcept nogil

cdef void Compute_grad_velocities_vectorized(
    double* mom         , double* grad_mom  , double* res           ,
    Py_ssize_t nbin     , Py_ssize_t geodim ,
    Py_ssize_t nsegm    , Py_ssize_t nvec   , Py_ssize_t grad_ndof  ,
    double* InvSegmMass , 
) noexcept nogil

cdef void Compute_velocities_user_data(
    double t    , double[::1] mom   , double[::1] res   , void* user_data   ,
) noexcept nogil

cdef void Compute_grad_velocities_user_data(
    double t    , double[::1] mom   , double[:,::1] grad_mom  , double[:,::1] res   , void* user_data   ,
) noexcept nogil

cdef void Compute_velocities_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_mom , double[:,::1] all_res , void* user_data   ,
) noexcept nogil

cdef void Compute_grad_velocities_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_mom , double[:,:,::1] all_grad_mom , double[:,:,::1] all_res , void* user_data   ,
) noexcept nogil
