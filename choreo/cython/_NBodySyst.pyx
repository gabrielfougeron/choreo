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

import choreo.metadata

cimport scipy.linalg.cython_blas
from choreo.scipy_plus.cython.blas_consts cimport *
from choreo.scipy_plus.cython.ccallback cimport ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS, ccallback_signature_t
from choreo.cython._ActionSym cimport ActionSym

import ctypes

# Explicit imports to avoid mysterious problems with CCALLBACK_DEFAULTS
from choreo.NBodySyst_build import (
    ContainsDoubleEdges                 ,
    ContainsSelfReferringTimeRevSegment ,
    Build_BodyGraph                     ,
    Build_SegmGraph_NoPb                ,
    AccumulateBodyConstraints           ,
    AccumulateSegmentConstraints        ,
    AccumulateInstConstraints           ,
    ComputeParamBasis_InitVal           ,
    ComputeParamBasis_Loop              ,
    reorganize_All_params_basis         ,
    PlotTimeBodyGraph                   ,
    CountSegmentBinaryInteractions      ,
    BundleListOfShapes                  ,
    BundleListOfArrays                  ,
    Populate_allsegmpos                 ,
    AccumulateSegmSourceToTargetSym     ,
    FindAllBinarySegments               ,
    ReorganizeBinarySegments            ,
)

import math
import scipy
import networkx
import json
import types
import itertools
import functools
import inspect
import time

try:
    from matplotlib import pyplot as plt
    from matplotlib import colormaps
except:
    pass

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

from choreo import NUMBA_AVAILABLE
if NUMBA_AVAILABLE:
    from choreo.numba_funs import jit_inter_law, jit_inter_law_str

from choreo.cython.optional_pyfftw cimport pyfftw
from choreo.optional_pyfftw import p_pyfftw, PYFFTW_AVAILABLE

cdef int USE_SCIPY_FFT = 0
cdef int USE_MKL_FFT = 1
cdef int USE_FFTW_FFT = 2
cdef int USE_DUCC_FFT = 3

cdef int GENERAL_SYM = 0
cdef int RFFT = 1

shortcut_name = {
    GENERAL_SYM : "general_sym" ,
    RFFT        : "rfft"        ,
}

cdef ccallback_signature_t signatures[2]

ctypedef void (*inter_law_fun_type)(double, double*, void*) noexcept nogil 
signatures[0].signature = b"void (double, double *, void *)"
signatures[0].value = 0
signatures[1].signature = NULL

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
    inter_law_fun_type  inter_law   
    void*               inter_law_param_ptr

@cython.auto_pickle(False)
@cython.final
cdef class NBodySyst():
    """
    This class defines a N-body system
    """

    cdef readonly Py_ssize_t geodim
    """ :class:`python:int` Dimension of ambient space.
    
    Read-only attribute.
    """

    cdef readonly Py_ssize_t nbody
    """ :class:`python:int` Number of bodies in system.
    
    Read-only attribute.
    """

    cdef readonly Py_ssize_t nint_min
    """ :class:`python:int` Minimum number of integration points per period.
    
    Read-only attribute.
    """

    cdef readonly Py_ssize_t nloop
    """ :class:`python:int` Number of loops in system.
    
    Read-only attribute.
    """

    cdef readonly Py_ssize_t nsegm
    """ :class:`python:int` Number of segments in system.
    
    Read-only attribute.
    """

    cdef readonly Py_ssize_t nbin_segm_tot
    """ :class:`python:int` Total number of binary interactions between segments.
    
    Read-only attribute.
    """

    cdef readonly Py_ssize_t nbin_segm_unique
    """ :class:`python:int` Number of **unique** binary interactions between segments.
    
    Read-only attribute.
    """

    cdef readonly bint RequiresGreaterNStore
    """ :class:`python:bool` Whether the symmetry constraints require the rightmost integration point to be included in segments.

    See Also
    --------
    
    * :attr:`GreaterNStore`
    * :attr:`ForceGreaterNStore`
    * :attr:`segm_size`
    * :attr:`segm_store`

    Read-only attribute.
    """

    cdef readonly bint GreaterNStore
    """ :class:`python:bool` Whether the rightmost integration point is included in segments.

     If :attr:`GreaterNStore` is True, then :attr:`segm_store` equals :attr:`segm_size` + 1, otherwise :attr:`segm_store` equals :attr:`segm_size`.

    See Also
    --------

    * :attr:`RequiresGreaterNStore`
    
    Read-only attribute.
    """

    @property
    def ForceGreaterNStore(self):
        """ :class:`python:bool` Whether to force the rightmost integration point to be included in segments.
        
        .. note:: The symmetries constraints might require the rightmost integration point to be included in segments. In this case, :attr:`ForceGreaterNStore` is ignored.

        See Also
        --------

        * :attr:`GreaterNStore`
        * :attr:`RequiresGreaterNStore`
        * :attr:`segm_size`
        * :attr:`segm_store`

        """
        return self.GreaterNStore and (not self.RequiresGreaterNStore)
    
    @ForceGreaterNStore.setter
    @cython.final
    def ForceGreaterNStore(self, bint force_in):

        cdef bint NewGreaterNStore = force_in or self.RequiresGreaterNStore
        
        if NewGreaterNStore != self.GreaterNStore:
            self.GreaterNStore = NewGreaterNStore
            self.nint = self._nint

    cdef int _fft_backend

    cdef public object fftw_planner_effort
    cdef public object fftw_nthreads
    cdef public bint fftw_wisdom_only

    @property
    def fft_backend(self):
        """ :class:`python:str` Name of the FFT backend currently in use.
            
        Possible values are:

        * "scipy": Use the SciPy FFT implementation.
        * "mkl": Use the Intel MKL FFT implementation, if available.
        * "ducc": Use the DUCC FFT implementation, if available.
        * "fftw": Use the FFTW FFT implementation, if available.

        .. note:: Setting this property will trigger a re-initialization of the FFT backend.

        Raises:
            ValueError: If an invalid FFT backend is provided or if the required package for the specified backend is not available.

        """
        
        if self._fft_backend == USE_SCIPY_FFT:
            return "scipy"
        elif self._fft_backend == USE_MKL_FFT:
            return "mkl"
        elif self._fft_backend == USE_DUCC_FFT:
            return "ducc"
        elif self._fft_backend == USE_FFTW_FFT:
            return "fftw"
        else:
            raise ValueError("This error should never be triggered. This is a bug.")

    @fft_backend.setter
    @cython.final
    def fft_backend(self, backend):

        self.free_owned_memory()

        if backend == "scipy":
            self._fft_backend = USE_SCIPY_FFT
        
        elif backend == "mkl":
            if MKL_FFT_AVAILABLE:
                self._fft_backend = USE_MKL_FFT
            else:
                raise ValueError("The package mkl_fft could not be loaded. Please check your local install.")
        
        elif backend == "ducc":
            if DUCC_FFT_AVAILABLE:
                self._fft_backend = USE_DUCC_FFT
            else:
                raise ValueError("The package ducc0 could not be loaded. Please check your local install.")
        
        elif backend == "fftw":
            if PYFFTW_AVAILABLE:
                self._fft_backend = USE_FFTW_FFT
            else:
                raise ValueError("The package pyfftw could not be loaded. Please check your local install.")
        else:
            raise ValueError('Invalid FFT backend. Possible options are "scipy", "mkl", "ducc" or "fftw".')

        if self._nint_fac > 0:
            self.allocate_owned_memory()

    cdef Py_ssize_t[::1] _loopnb
    @property
    def loopnb(self):
        return np.asarray(self._loopnb)

    cdef Py_ssize_t[::1] _bodyloop
    @property
    def bodyloop(self):
        return np.asarray(self._bodyloop)

    cdef double[::1] _loopmass
    @property
    def loopmass(self):
        return np.asarray(self._loopmass)

    cdef double[::1] _invsegmmass
    cdef double[::1] _segmcharge

    cdef double[::1] _loopcharge
    @property
    def loopcharge(self):
        return np.asarray(self._loopcharge)

    cdef Py_ssize_t[:,::1] _Targets
    @property
    def Targets(self):
        return np.asarray(self._Targets)

    cdef Py_ssize_t[:,::1] _bodysegm
    @property
    def bodysegm(self):
        return np.asarray(self._bodysegm)

    cdef Py_ssize_t[::1] _loopgen
    @property
    def loopgen(self):
        return np.asarray(self._loopgen)

    cdef Py_ssize_t[::1] _intersegm_to_body
    @property
    def intersegm_to_body(self):
        return np.asarray(self._intersegm_to_body)

    cdef Py_ssize_t[::1] _intersegm_to_iint
    @property
    def intersegm_to_iint(self):
        return np.asarray(self._intersegm_to_iint)

    cdef Py_ssize_t[::1] _gensegm_to_body
    @property
    def gensegm_to_body(self):
        return np.asarray(self._gensegm_to_body)

    cdef Py_ssize_t[::1] _gensegm_to_iint
    @property
    def gensegm_to_iint(self):
        return np.asarray(self._gensegm_to_iint)

    cdef Py_ssize_t[::1] _gensegm_to_iintrel
    @property
    def gensegm_to_iintrel(self):
        return np.asarray(self._gensegm_to_iintrel)

    cdef Py_ssize_t[::1] _ngensegm_loop
    @property
    def ngensegm_loop(self):
        return np.asarray(self._ngensegm_loop)

    cdef Py_ssize_t[::1] _gensegm_loop_start
    @property
    def gensegm_loop_start(self):
        return np.asarray(self.gensegm_loop_start)

    cdef Py_ssize_t[::1] _n_sub_fft
    @property
    def n_sub_fft(self):
        return np.asarray(self._n_sub_fft)

    cdef Py_ssize_t[::1] _BinSourceSegm
    @property
    def BinSourceSegm(self):
        return np.asarray(self._BinSourceSegm)

    cdef Py_ssize_t[::1] _BinTargetSegm
    @property
    def BinTargetSegm(self):
        return np.asarray(self._BinTargetSegm)
        
    cdef double[:,:,::1] _BinSpaceRot
    @property
    def BinSpaceRot(self):
        return np.asarray(self._BinSpaceRot)        

    cdef bint[::1] _BinSpaceRotIsId
    @property
    def BinSpaceRotIsId(self):
        return np.asarray(self._BinSpaceRotIsId) > 0
        
    cdef double[::1] _BinProdChargeSum
    @property
    def BinProdChargeSum(self):
        return np.asarray(self._BinProdChargeSum)

    cdef Py_ssize_t[::1] _InterTimeRev
    @property
    def InterTimeRev(self):
        return np.asarray(self._InterTimeRev)

    cdef bint[::1] _InterSpaceRotPosIsId
    @property
    def InterSpaceRotIsId(self):
        return np.asarray(self._InterSpaceRotPosIsId) > 0

    cdef double[:,:,::1] _InterSpaceRotPos
    @property
    def InterSpaceRot(self):
        return np.asarray(self._InterSpaceRotPos)
        
    cdef bint[::1] _InterSpaceRotVelIsId
    @property
    def InterSpaceRotIsId(self):
        return np.asarray(self._InterSpaceRotVelIsId) > 0

    cdef double[:,:,::1] _InterSpaceRotVel
    @property
    def InterSpaceRot(self):
        return np.asarray(self._InterSpaceRotVel)

    # ALG for After Last Gen => interpolation for 1 point mismatch between segm_size and segm_store
    cdef Py_ssize_t[::1] _ALG_Iint
    @property
    def ALG_Iint(self):
        return np.asarray(self._ALG_Iint)

    cdef Py_ssize_t[::1] _ALG_TimeRev
    @property
    def ALG_TimeRev(self):
        return np.asarray(self._ALG_TimeRev)

    cdef double[:,:,::1] _ALG_SpaceRotPos
    @property
    def ALG_SpaceRot(self):
        return np.asarray(self._ALG_SpaceRotPos)

    cdef double[:,:,::1] _ALG_SpaceRotVel
    @property
    def ALG_SpaceRot(self):
        return np.asarray(self._ALG_SpaceRotVel)

    cdef Py_ssize_t[::1] _PerDefBeg_Isegm
    @property
    def PerDefBeg_Isegm(self):
        return np.asarray(self._PerDefBeg_Isegm)

    cdef Py_ssize_t[::1] _PerDefBeg_TimeRev
    @property
    def PerDefBeg_TimeRev(self):
        return np.asarray(self._PerDefBeg_TimeRev)

    cdef double[:,:,::1] _PerDefBeg_SpaceRotPos
    @property
    def PerDefBeg_SpaceRotPos(self):
        return np.asarray(self._PerDefBeg_SpaceRotPos)

    cdef double[:,:,::1] _PerDefBeg_SpaceRotVel
    @property
    def PerDefBeg_SpaceRotVel(self):
        return np.asarray(self._PerDefBeg_SpaceRotVel)

    cdef Py_ssize_t[::1] _PerDefEnd_Isegm
    @property
    def PerDefEnd_Isegm(self):
        return np.asarray(self._PerDefEnd_Isegm)

    cdef Py_ssize_t[::1] _PerDefEnd_TimeRev
    @property
    def PerDefEnd_TimeRev(self):
        return np.asarray(self._PerDefEnd_TimeRev)

    cdef double[:,:,::1] _PerDefEnd_SpaceRotPos
    @property
    def PerDefEnd_SpaceRotPos(self):
        return np.asarray(self._PerDefEnd_SpaceRotPos)

    cdef double[:,:,::1] _PerDefEnd_SpaceRotVel
    @property
    def PerDefEnd_SpaceRotVel(self):
        return np.asarray(self._PerDefEnd_SpaceRotVel)

    cdef double[:,:,::1] _InitValPosBasis
    @property
    def InitValPosBasis(self):
        return np.asarray(self._InitValPosBasis)

    cdef double[:,:,::1] _InitValVelBasis
    @property
    def InitValVelBasis(self):
        return np.asarray(self._InitValVelBasis)
        
    cdef double complex[::1] _params_basis_buf_pos
    cdef double complex[::1] _params_basis_buf_vel
    cdef Py_ssize_t[:,::1] _params_basis_shapes
    cdef Py_ssize_t[::1] _params_basis_shifts

    def params_basis_pos(self, Py_ssize_t il):
        return np.asarray(self._params_basis_buf_pos[self._params_basis_shifts[il]:self._params_basis_shifts[il+1]]).reshape(self._params_basis_shapes[il])
        
    def params_basis_vel(self, Py_ssize_t il):
        return np.asarray(self._params_basis_buf_vel[self._params_basis_shifts[il]:self._params_basis_shifts[il+1]]).reshape(self._params_basis_shapes[il])

    cdef Py_ssize_t[::1] _nnz_k_buf
    cdef Py_ssize_t[:,::1] _nnz_k_shapes
    cdef Py_ssize_t[::1] _nnz_k_shifts

    def nnz_k(self, Py_ssize_t il):
        return np.asarray(self._nnz_k_buf[self._nnz_k_shifts[il]:self._nnz_k_shifts[il+1]]).reshape(self._nnz_k_shapes[il])

    # Removal of imaginary part of c_o
    cdef bint[::1] _co_in_buf
    cdef Py_ssize_t[:,::1] _co_in_shapes
    cdef Py_ssize_t[::1] _co_in_shifts
    cdef Py_ssize_t[::1] _ncor_loop
    cdef Py_ssize_t[::1] _nco_in_loop
    cdef readonly Py_ssize_t nrem

    def co_in(self, Py_ssize_t il):
        return np.asarray(self._co_in_buf[self._co_in_shifts[il]:self._co_in_shifts[il+1]]).reshape(self._co_in_shapes[il]) > 0

    cdef Py_ssize_t[::1] _ncoeff_min_loop
    @property
    def ncoeff_min_loop(self):
        return np.asarray(self._ncoeff_min_loop)

    cdef readonly list Sym_list
    cdef readonly object BodyGraph
    cdef readonly object SegmGraph

    cdef readonly list gensegm_to_all
    cdef readonly list intersegm_to_all
    cdef readonly list LoopGenConstraints

    cdef bint[:,::1] _SegmRequiresDisp
    @property
    def SegmRequiresDisp(self):
        return np.asarray(self._SegmRequiresDisp) > 0

    cdef double[::1] _Hash_exp
    @property
    def Hash_exp(self):
        return np.asarray(self._Hash_exp)

    cdef inter_law_fun_type _inter_law
    cdef readonly str inter_law_str
    """ :class:`python:str` A description of the interaction law"""

    cdef readonly object inter_law_param_dict
    cdef double[::1] _inter_law_param_buf
    cdef void* _inter_law_param_ptr
    
    cdef readonly bint LawIsHomo
    cdef readonly double Homo_exp
    cdef readonly double Homo_unit

    # Things that change with nint
    cdef Py_ssize_t _nint
    @property
    def nint(self):
        """ :class:`python:int` The number of integration points per period.

        Raises:
            ValueError: If the provided :attr:`nint` is not divisible by :math:`2 *` :attr:`nint_min`.
        """

        return self._nint

    @nint.setter
    @cython.cdivision(True)
    @cython.final
    def nint(self, Py_ssize_t nint_in):

        if (nint_in % (2 * self.nint_min)) != 0:
            raise ValueError(f"Provided nint {nint_in} should be divisible by {2 * self.nint_min}")

        self._nint = nint_in
        self._nint_fac = nint_in // (2 * self.nint_min)
        self.ncoeffs = self._nint // 2 + 1
        self.segm_size = self._nint // self.nint_min

        if self.GreaterNStore:
            self.segm_store = self.segm_size + 1
        else:
            self.segm_store = self.segm_size

        params_shapes_list = []
        ifft_shapes_list = []
        pos_slice_shapes_list = []
        for il in range(self.nloop):

            nppl = self._params_basis_shapes[il,2]
            assert self._nint % (2*self._ncoeff_min_loop[il]) == 0
            npr = self._nint //  (2*self._ncoeff_min_loop[il])
            
            params_shapes_list.append((npr, self._nnz_k_shapes[il,0], nppl))
            ifft_shapes_list.append((npr+1, self._nnz_k_shapes[il,0], nppl))
            
            if self._n_sub_fft[il] == 2:
                ninter = npr+1
            elif self._n_sub_fft[il] == 1:
                if self.GreaterNStore: 
                    ninter = 2*npr+1
                else:
                    ninter = 2*npr
            else:
                raise ValueError(f'Impossible value for n_sub_fft[il]: {self._n_sub_fft[il]}. Allowed values are either 1 or 2.')   

            pos_slice_shapes_list.append((ninter, self.geodim))
            
        self._params_shapes, self._params_shifts = BundleListOfShapes(params_shapes_list)
        self._ifft_shapes, self._ifft_shifts = BundleListOfShapes(ifft_shapes_list)
        self._pos_slice_shapes, self._pos_slice_shifts = BundleListOfShapes(pos_slice_shapes_list)

        self.nparams_incl_o = 2*self._params_shifts[self.nloop]
        self.nparams = self._params_shifts[self.nloop] - self.nrem

        self.free_owned_memory()
        self.allocate_owned_memory()

    cdef Py_ssize_t _nint_fac
    @property
    def nint_fac(self):
        """ :class:`python:int` Half the size of a segment. Changing this value is the preferred way to change the number of integration points because there are no constraints on :attr:`segm_store` besides that it be a positive integer.
        """
        
        return self._nint_fac

    @nint_fac.setter
    @cython.final
    def nint_fac(self, Py_ssize_t nint_fac_in):
        self.nint = 2 * self.nint_min * nint_fac_in

    cdef readonly Py_ssize_t ncoeffs
    """ :class:`python:int` Number of Fourier coefficients in each loop.
    
    Read-only attribute.
    """

    cdef readonly Py_ssize_t segm_size    # number of interacting nodes in segment
    """ :class:`python:int` Number of interacting positions in a segment.
    
    .. note ::
        The value of :attr:`segm_size` can differ from that of :attr:`segm_store` if both endpoints are explicitly stored.

    See Also
    --------

    * :attr:`ForceGreaterNStore`

    Read-only attribute.
    """
    cdef readonly Py_ssize_t segm_store   # number of stored values in segment, including repeated values for n_sub_fft == 2
    """ :class:`python:int` Number of stored positions in a segment.

    Read-only attribute.

    .. note ::
        The value of :attr:`segm_size` can differ from that of :attr:`segm_store` if both endpoints are explicitly stored.

    See Also
    --------

    * :attr:`ForceGreaterNStore`

    """
    cdef readonly Py_ssize_t nparams
    """ :class:`python:int` Number of parameters.
    
    Read-only attribute.
    """
    
    cdef readonly Py_ssize_t nparams_incl_o
    """ :class:`python:int` Number of parameters after removal of unnecessary parts of zero-indexed Fourier coefficients.
    
    Read-only attribute.
    """

    # WARNING: These are the shapes and shifts of POS params, NOT MOM params!
    cdef Py_ssize_t[:,::1] _params_shapes   
    cdef Py_ssize_t[::1] _params_shifts

    cdef Py_ssize_t[:,::1] _ifft_shapes      
    cdef Py_ssize_t[::1] _ifft_shifts

    cdef Py_ssize_t[:,::1] _pos_slice_shapes
    cdef Py_ssize_t[::1] _pos_slice_shifts

    @property
    def params_shapes(self):
        return np.asarray(self._params_shapes)
    @property
    def ifft_shapes(self):
        return np.asarray(self._ifft_shapes)
    @property
    def pos_slice_shapes(self):
        return np.asarray(self._pos_slice_shapes)

    @property
    def params_shifts(self):
        return np.asarray(self._params_shifts)

    cdef bint BufArraysAllocated
    cdef double** _pos_slice_buf_ptr
    cdef double** _params_pos_buf
    cdef double complex** _ifft_buf_ptr

    def pos_slice(self, il):

        if self._pos_slice_buf_ptr == NULL:
            return None

        cdef Py_ssize_t _il = il
        cdef double[:,::1] res = <double[:self._pos_slice_shapes[_il,0],:self._pos_slice_shapes[_il,1]]> self._pos_slice_buf_ptr[_il]

        return np.asarray(res)

    cdef list _fftw_genrfft
    cdef pyfftw.fftw_exe** _fftw_genrfft_exe

    cdef list _fftw_genirfft
    cdef pyfftw.fftw_exe** _fftw_genirfft_exe

    cdef list _fftw_symrfft
    cdef pyfftw.fftw_exe** _fftw_symrfft_exe

    cdef list _fftw_symirfft
    cdef pyfftw.fftw_exe** _fftw_symirfft_exe

    cdef list _fftw_arrays

    cdef double[:,:,::1] _segmpos 
    cdef double[:,:,::1] _pot_nrg_grad

    cdef bint _ForceGeneralSym
    @property
    def ForceGeneralSym(self):
        return self._ForceGeneralSym

    @ForceGeneralSym.setter
    @cython.cdivision(True)
    @cython.final
    def ForceGeneralSym(self, bint force_in):

        self.free_owned_memory()

        self._ForceGeneralSym = force_in

        if force_in:
            self._ParamBasisShortcutPos = np.full((self.nloop), GENERAL_SYM, dtype=np.intc)
            self._ParamBasisShortcutVel = np.full((self.nloop), GENERAL_SYM, dtype=np.intc)
        else:
            self._ParamBasisShortcutPos = self._ParamBasisShortcutPos_th.copy()
            self._ParamBasisShortcutVel = self._ParamBasisShortcutVel_th.copy()

        self.allocate_owned_memory()

    cdef int[::1] _ParamBasisShortcutPos_th
    cdef int[::1] _ParamBasisShortcutVel_th

    cdef int[::1] _ParamBasisShortcutPos
    @property
    def ParamBasisShortcutPos(self):
        return [shortcut_name[shortcut] for shortcut in self._ParamBasisShortcutPos]

    cdef int[::1] _ParamBasisShortcutVel
    @property
    def ParamBasisShortcutVel(self):
        return [shortcut_name[shortcut] for shortcut in self._ParamBasisShortcutVel]

    cdef ODE_params_t _ODE_params 
    cdef void *_ODE_params_ptr

    def __init__(
        self                                ,
        Py_ssize_t geodim = 2               ,
        Py_ssize_t nbody = 2                ,
        double[::1] bodymass = None         ,
        double[::1] bodycharge  = None      ,
        list Sym_list = []                  ,
        object inter_law = None             , 
        str inter_law_str = None            , 
        object inter_law_param_dict = None  ,
        bint ForceGeneralSym = False        ,
        bint ForceGreaterNStore = False     ,
    ):
        """ Defines a N-Body System.

        See Also
        --------

        * :meth:`Set_inter_law`

        Parameters
        ----------
        geodim : :class:`python:int`, optional
            Number of dimensions of ambiant space. Typically 2 or 3, but can be any positive integer. By default: 2.
        nbody : :class:`python:int`, optional
            Number of bodies in the system, by default 2.
        bodymass :  :class:`numpy:numpy.ndarray`:class:`(shape = (nbody), dtype = np.float64)`, optional
            Masses of the bodies in the system, by default :obj:`numpy:numpy.ones`.
        bodycharge : :class:`numpy:numpy.ndarray`:class:`(shape = (nbody), dtype = np.float64)`, optional
            Charges of the bodies, by default :obj:`numpy:numpy.ones`.
        Sym_list : :class:`python:list`, optional
            List of a priori symmetries in the system, by default ``[]``.
        inter_law : optional
            Function defining the interaction law, by default :data:`python:None`.
        inter_law_str : :class:`python:str`, optional
            Description of the interaction law dictating the dynamics of the system, by default :data:`python:None`.
        inter_law_param_dict : :class:`python:dict`, optional
            Parameters pertaining to the interaction law, by default :data:`python:None`.
        ForceGeneralSym : :class:`python:bool`, optional
            Whether to force the symmetries to be treated in full generality when computing positions or velocities from parameters (and vice-versa, both in direct or adjoint mode), or to try an optimized route instead. Most users should leave this option to its default value, which is :data:`python:False`.
        ForceGreaterNStore : :class:`python:bool`, optional
            Whether to force the number of stored segment positions to be increased, even though the symmetries might not require it. Most users should leave this option to its default value, which is :data:`python:False`.

        """    

        self._nint_fac = 0 
        self.BufArraysAllocated = False

        if geodim < 1:
            raise ValueError(f"geodim should be a positive integer. Received {geodim = }")
        self.geodim = geodim
        
        if nbody < 1:
            raise ValueError(f"nbody should be a positive integer. Received {nbody = }")
        self.nbody = nbody

        cdef Py_ssize_t i, il, ibin, ib
        cdef double eps = 1e-12
        cdef ActionSym Sym, Shift

        if bodymass is None:
            bodymass = np.ones((nbody), dtype = np.float64)
        if (bodymass.shape[0] != nbody):
            raise ValueError(f'Incompatible number of bodies {nbody} vs number of masses {bodymass.shape[0]}.')
        if bodycharge is None:
            bodycharge = np.ones((nbody), dtype = np.float64)
        if (bodycharge.shape[0] != nbody):
            raise ValueError(f'Incompatible number of bodies {nbody} vs number of charges {bodycharge.shape[0]}.')

        self.Set_inter_law(inter_law, inter_law_str, inter_law_param_dict)

        self._Hash_exp = default_Hash_exp


        self.Sym_list = Sym_list
        # Zero charges are OK but not zero masses
        for ib in range(nbody):
            assert bodymass[ib] != 0.

        self.RequiresGreaterNStore = False
        for Sym in self.Sym_list:
            self.RequiresGreaterNStore = self.RequiresGreaterNStore or (Sym.TimeRev < 0)

        self.DetectLoops(bodymass, bodycharge)
        self.BuildSegmGraph()
        self.ChooseInterSegm()
        self.intersegm_to_all = AccumulateSegmSourceToTargetSym(self.SegmGraph, nbody, geodim, self.nint_min, self.nsegm, self._intersegm_to_iint, self._intersegm_to_body)
    
        self.ChooseLoopGen()
        self.gensegm_to_all = AccumulateSegmSourceToTargetSym(self.SegmGraph, nbody, geodim, self.nint_min, self.nsegm, self._gensegm_to_iint, self._gensegm_to_body)
        self.GatherInterSym()

        # self.SegmConstraints = AccumulateSegmentConstraints(self.SegmGraph, nbody, geodim, self.nsegm, self._bodysegm)

        # Setting up forward ODE:
        # - What are my parameters ?
        # - Integration end + Lack of periodicity
        # - Constraints on initial values => Parametrization 
#         
#         InstConstraintsPos = AccumulateInstConstraints(self.Sym_list, nbody, geodim, self.nint_min, VelSym=False)
#         InstConstraintsVel = AccumulateInstConstraints(self.Sym_list, nbody, geodim, self.nint_min, VelSym=True )
# 
#         self._InitValPosBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsPos[0], bodymass, MomCons=True)
#         self._InitValVelBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsVel[0], bodymass, MomCons=True)

        BinarySegm = FindAllBinarySegments(self.intersegm_to_all, nbody, self.nsegm, self.nint_min, self._bodysegm, bodycharge)
        self.nbin_segm_tot, self.nbin_segm_unique = CountSegmentBinaryInteractions(BinarySegm, self.nsegm)

        self._BinSourceSegm, self._BinTargetSegm, BinTimeRev, self._BinSpaceRot, self._BinProdChargeSum = ReorganizeBinarySegments(BinarySegm)

        # Not actually sure this is always true.
        if not (BinTimeRev == 1).all():
            raise ValueError("Catastrophic failure: BinTimeRev not as expected")

        assert self._BinSourceSegm.shape[0] == self.nbin_segm_unique
        self._BinSpaceRotIsId = np.zeros((self.nbin_segm_unique), dtype=np.intc)
        for ibin in range(self.nbin_segm_unique):
            self._BinSpaceRotIsId[ibin] = (np.linalg.norm(self._BinSpaceRot[ibin,:,:] - np.identity(self.geodim)) < eps)
            self._BinProdChargeSum[ibin] /= self.nint_min

        self.DetectSegmRequiresDisp()

        # This could certainly be made more efficient
        BodyConstraints = AccumulateBodyConstraints(self.Sym_list, nbody, geodim)
        self.LoopGenConstraints = [BodyConstraints[ib] for ib in self._loopgen]

        ShiftedLoopGenConstraints = []
        for il in range(self.nloop):

            ib = self._loopgen[il]

            ShiftedBodyConstraints = []
            Shift = ActionSym(
                BodyPerm  = np.array(range(nbody), dtype = np.intp) ,
                SpaceRot  = np.identity(geodim, dtype = np.float64) ,
                TimeRev   = 1                                       ,
                TimeShiftNum = - self._gensegm_loop_start[il]       ,
                TimeShiftDen = self.nint_min                        ,
            )

            for Sym in BodyConstraints[ib]:
                ShiftedBodyConstraints.append(Sym.Conjugate(Shift))

            ShiftedLoopGenConstraints.append(ShiftedBodyConstraints)

        # Idem, but I'm too lazy to change it and it is not performance critical
        All_params_basis_pos = ComputeParamBasis_Loop(self.nloop, self._loopgen, geodim, ShiftedLoopGenConstraints)

        self._ncoeff_min_loop = np.array([len(All_params_basis_pos[il]) for il in range(self.nloop)], dtype=np.intp)
        params_basis_reorganized_list, nnz_k_list, co_in_list = reorganize_All_params_basis(All_params_basis_pos)
        self._params_basis_buf_pos, self._params_basis_shapes, self._params_basis_shifts = BundleListOfArrays(params_basis_reorganized_list)

        self._params_basis_buf_vel = np.empty(self._params_basis_buf_pos.shape[0], dtype=np.complex128)
        for i in range(self._params_basis_buf_pos.shape[0]):
            self._params_basis_buf_vel[i] = self._params_basis_buf_pos[i] * 1j

        self._nnz_k_buf, self._nnz_k_shapes, self._nnz_k_shifts = BundleListOfArrays(nnz_k_list)
        self._co_in_buf, self._co_in_shapes, self._co_in_shifts = BundleListOfArrays(co_in_list)

        if self._nnz_k_shifts[self.nloop] == 0:
            raise ValueError("Provided symmetries resulted in an empty parameter basis.")

        self.ConfigureShortcutSym()

        self._nco_in_loop = np.zeros((self.nloop), dtype=np.intp)
        self._ncor_loop = np.zeros((self.nloop), dtype=np.intp)
        for il in range(self.nloop):
            for i in range(self._co_in_shifts[il], self._co_in_shifts[il+1]):
                if self._co_in_buf[i]:
                    self._ncor_loop[il] +=1
                else:
                    self._nco_in_loop[il] +=1

        self.nrem = np.sum(self._nco_in_loop)

        self.Compute_n_sub_fft()

        self._invsegmmass = np.empty((self.nsegm), dtype=np.float64)
        self._segmcharge = np.empty((self.nsegm), dtype=np.float64)
        for isegm in range(self.nsegm):
            self._invsegmmass[isegm] = 1. / self._loopmass[self._bodyloop[self._intersegm_to_body[isegm]]]
            self._segmcharge[isegm] = self._loopcharge[self._bodyloop[self._intersegm_to_body[isegm]]]

        self.Update_ODE_params()

        if MKL_FFT_AVAILABLE:
            self.fft_backend = "mkl"
        else:
            self.fft_backend = "scipy"

        self.fftw_planner_effort = 'FFTW_ESTIMATE'
        self.fftw_nthreads = 1
        self.fftw_wisdom_only = False

        self.GreaterNStore = self.RequiresGreaterNStore or ForceGreaterNStore
        self.nint_fac = 1
        self.ForceGeneralSym = ForceGeneralSym

    def __dealloc__(self):
        self.free_owned_memory()

    @cython.final
    def free_owned_memory(self):
        """ Frees the memory allocated for the buffer arrays used in :class:`NBodySyst`.
        """
        
        cdef Py_ssize_t il

        if self.BufArraysAllocated:

            if self.fft_backend in ['scipy', 'mkl', 'ducc']:
                
                for il in range(self.nloop):
                    free(self._pos_slice_buf_ptr[il])
                    free(self._params_pos_buf[il])
                    free(self._ifft_buf_ptr[il])
            
            elif self.fft_backend in ['fftw']:

                self._fftw_genrfft = None
                self._fftw_genirfft = None
                self._fftw_symrfft = None
                self._fftw_symirfft = None

                for il in range(self.nloop):
                    
                    free(self._fftw_genrfft_exe[il])
                    free(self._fftw_genirfft_exe[il])

                for il in range(self.nloop):
                    if self._fftw_symrfft_exe[il] != NULL:
                        free(self._fftw_symrfft_exe[il])
                        self._fftw_symrfft_exe[il] = NULL

                    if self._fftw_symirfft_exe[il] != NULL:
                        free(self._fftw_symirfft_exe[il])
                        self._fftw_symirfft_exe[il] = NULL

                free(self._fftw_genrfft_exe)
                free(self._fftw_genirfft_exe)
                free(self._fftw_symrfft_exe)
                free(self._fftw_symirfft_exe)

            free(self._pos_slice_buf_ptr)
            free(self._params_pos_buf)
            free(self._ifft_buf_ptr)

        self.BufArraysAllocated = False

    @cython.final
    def allocate_owned_memory(self):
        """ Allocates the buffer arrays used in :class:`NBodySyst` depending on the value of :attr:`nint` and :attr:`fft_backend`.
        """
        
        cdef Py_ssize_t il

        self._pos_slice_buf_ptr = <double**> malloc(sizeof(double*)*self.nloop)
        self._params_pos_buf = <double**> malloc(sizeof(double*)*self.nloop)
        self._ifft_buf_ptr = <double complex**> malloc(sizeof(double complex**)*self.nloop)
        cdef double *pyfftw_input_array_start
        cdef double complex *pyfftw_output_array_start

        cdef pyfftw.FFTW pyfftw_object
        cdef double[:,:,::1] pyfftw_input_array
        cdef double complex[:,:,::1] pyfftw_output_array

        cdef double complex[:,::1] pyfftw_input_array_2
        cdef double[:,::1] pyfftw_output_array_2

        cdef double[::1] pos_slice_1d
        cdef double[:,::1] pos_slice_mv
        cdef double complex[:,::1] params_c_mv

        if self.fft_backend in ['scipy', 'mkl', 'ducc']:

            for il in range(self.nloop):
                self._pos_slice_buf_ptr[il] = <double*> malloc(sizeof(double)*(self._pos_slice_shifts[il+1]-self._pos_slice_shifts[il]))
                self._params_pos_buf[il] = <double*> malloc(sizeof(double)*2*(self._params_shifts[il+1]-self._params_shifts[il]))
                self._ifft_buf_ptr[il] = <double complex*> malloc(sizeof(double complex)*(self._ifft_shifts[il+1]-self._ifft_shifts[il]))

        elif self.fft_backend in ['fftw']:

            self._fftw_genrfft = [None for _ in range(self.nloop)]
            self._fftw_genirfft = [None for _ in range(self.nloop)]
            self._fftw_symrfft = [None for _ in range(self.nloop)]
            self._fftw_symirfft = [None for _ in range(self.nloop)]
            self._fftw_arrays = []

            self._fftw_genrfft_exe = <pyfftw.fftw_exe**> malloc(sizeof(pyfftw.fftw_exe*) * self.nloop)
            self._fftw_genirfft_exe = <pyfftw.fftw_exe**> malloc(sizeof(pyfftw.fftw_exe*) * self.nloop)
            self._fftw_symrfft_exe = <pyfftw.fftw_exe**> malloc(sizeof(pyfftw.fftw_exe*) * self.nloop)
            self._fftw_symirfft_exe = <pyfftw.fftw_exe**> malloc(sizeof(pyfftw.fftw_exe*) * self.nloop)

            for il in range(self.nloop):

                params_pos = p_pyfftw.empty_aligned((2*self._params_shapes[il,0],self._params_shapes[il,1],self._params_shapes[il,2]), dtype=np.float64)
                ifft = p_pyfftw.empty_aligned((self._ifft_shapes[il,0],self._ifft_shapes[il,1],self._ifft_shapes[il,2]), dtype=np.complex128)

                flags = [self.fftw_planner_effort, 'FFTW_DESTROY_INPUT']

                if self.fftw_wisdom_only:
                    flags.append('FFTW_WISDOM_ONLY')

                direction = 'FFTW_FORWARD'
                pyfftw_object = pyfftw.FFTW(params_pos, ifft, axes=(0, ), direction=direction, flags=flags, threads=self.fftw_nthreads)     
                self._fftw_genrfft[il] = pyfftw_object

                pyfftw_input_array = pyfftw_object.input_array
                if self._params_shapes[il,1] > 0:
                    pyfftw_input_array_start = &pyfftw_input_array[0,0,0]
                else:
                    pyfftw_input_array_start = NULL
                self._params_pos_buf[il] = pyfftw_input_array_start

                pyfftw_output_array = pyfftw_object.output_array
                if self._params_shapes[il,1] > 0:
                    pyfftw_output_array_start = &pyfftw_output_array[0,0,0]
                else:
                    pyfftw_output_array_start = NULL
                self._ifft_buf_ptr[il] = pyfftw_output_array_start

                self._fftw_genrfft_exe[il] = <pyfftw.fftw_exe*> malloc(sizeof(pyfftw.fftw_exe))
                self._fftw_genrfft_exe[il][0] = pyfftw_object.get_fftw_exe()

                direction = 'FFTW_BACKWARD'
                pyfftw_object = pyfftw.FFTW(ifft, params_pos, axes=(0, ), direction=direction, flags=flags, threads=self.fftw_nthreads)   
                self._fftw_genirfft[il] = pyfftw_object

                self._fftw_genirfft_exe[il] = <pyfftw.fftw_exe*> malloc(sizeof(pyfftw.fftw_exe))
                self._fftw_genirfft_exe[il][0] = pyfftw_object.get_fftw_exe()

                if (self._ParamBasisShortcutPos_th[il] == RFFT) or (self._ParamBasisShortcutVel_th[il] == RFFT):

                    pyfftw_input_array_2 = <double complex[:(self._params_shapes[il,0]+1),:self.geodim]> (<double complex*> pyfftw_input_array_start)

                    pos_slice_1d = p_pyfftw.empty_aligned((self._pos_slice_shifts[il+1]-self._pos_slice_shifts[il]), dtype=np.float64)
                    self._pos_slice_buf_ptr[il] = &pos_slice_1d[0]

                    pyfftw_output_array_2 = <double[:2*self._params_shapes[il,0],:self.geodim:1]> self._pos_slice_buf_ptr[il]

                    direction = 'FFTW_BACKWARD'
                    pyfftw_object = pyfftw.FFTW(np.asarray(pyfftw_input_array_2), np.asarray(pyfftw_output_array_2), axes=(0, ), direction=direction, flags=flags, threads=self.fftw_nthreads)  
                    self._fftw_arrays.append(pos_slice_1d)

                    self._fftw_symirfft[il] = pyfftw_object

                    self._fftw_symirfft_exe[il] = <pyfftw.fftw_exe*> malloc(sizeof(pyfftw.fftw_exe))
                    self._fftw_symirfft_exe[il][0] = pyfftw_object.get_fftw_exe()

                    n = 2*(self._ifft_shapes[il,0] - 1)
                    pos_slice_mv = <double[:n,:self.geodim:1]> self._pos_slice_buf_ptr[il]
                    params_c_mv = <double complex[:self._ifft_shapes[il,0],:self.geodim:1]> (<double complex*> self._params_pos_buf[il])

                    direction = 'FFTW_FORWARD'
                    pyfftw_object = pyfftw.FFTW(np.asarray(pos_slice_mv), np.asarray(params_c_mv), axes=(0, ), direction=direction, flags=flags, threads=self.fftw_nthreads)  
                    self._fftw_symrfft[il] = pyfftw_object

                    self._fftw_symrfft_exe[il] = <pyfftw.fftw_exe*> malloc(sizeof(pyfftw.fftw_exe))
                    self._fftw_symrfft_exe[il][0] = pyfftw_object.get_fftw_exe()

                else:

                    self._fftw_symirfft_exe[il] = NULL
                    self._fftw_symrfft_exe[il]  = NULL

                    self._pos_slice_buf_ptr[il] = <double*> malloc(sizeof(double)*(self._pos_slice_shifts[il+1]-self._pos_slice_shifts[il]))

        self._segmpos = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)
        self._pot_nrg_grad = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)

        self.BufArraysAllocated = True

    @cython.final
    def Update_ODE_params(self):
        
        self._ODE_params.nbin                       =  self.nbin_segm_unique
        self._ODE_params.geodim                     =  self.geodim
        self._ODE_params.nsegm                      =  self.nsegm
        self._ODE_params.InvSegmMass                = &self._invsegmmass[0]
        self._ODE_params.SegmCharge                 = &self._segmcharge[0]
        self._ODE_params.BinSourceSegm_ptr          = &self._BinSourceSegm[0]
        self._ODE_params.BinTargetSegm_ptr          = &self._BinTargetSegm[0]
        self._ODE_params.BinSpaceRot_ptr            = &self._BinSpaceRot[0,0,0]
        self._ODE_params.BinSpaceRotIsId_ptr        = &self._BinSpaceRotIsId[0]
        self._ODE_params.BinProdChargeSum_ptr       = &self._BinProdChargeSum[0]
        self._ODE_params.inter_law                  =  self._inter_law
        self._ODE_params.inter_law_param_ptr        =  self._inter_law_param_ptr

        self._ODE_params_ptr = <void*> &self._ODE_params

    @cython.final
    def Get_ODE_params(self):
        """ Retrieve a pointer to ODE parameters definition
        """

        self.Update_ODE_params()
        user_data = ctypes.c_void_p(<uintptr_t> self._ODE_params_ptr)

        return user_data

    @cython.final
    def DetectLoops(self, double[::1] bodymass, double[::1] bodycharge, Py_ssize_t nint_min_fac = 1):

        cdef Py_ssize_t il, ib, ilb
        
        All_den_list_on_entry = []
        for Sym in self.Sym_list:
            All_den_list_on_entry.append(Sym.TimeShiftDen)

        self.nint_min = nint_min_fac * math.lcm(*All_den_list_on_entry) # ensures that all integer divisions will have zero remainder
        
        BodyGraph = Build_BodyGraph(self.nbody, self.Sym_list)

        self.nloop = sum(1 for _ in networkx.connected_components(BodyGraph))
        
        loopnb = np.zeros((self.nloop), dtype = np.intp)
        self._loopnb = loopnb

        for il, CC in enumerate(networkx.connected_components(BodyGraph)):
            loopnb[il] = len(CC)

        maxlooplen = loopnb.max()
        
        BodyLoop = np.zeros((self.nbody), dtype = np.intp)
        self._bodyloop = BodyLoop
        cdef Py_ssize_t[:,::1] Targets = np.zeros((self.nloop, maxlooplen), dtype=np.intp)
        self._Targets = Targets
        for il, CC in enumerate(networkx.connected_components(BodyGraph)):
            for ilb, ib in enumerate(CC):
                Targets[il,ilb] = ib
                BodyLoop[ib] = il

        loopmass = np.zeros((self.nloop), dtype=np.float64)
        self._loopmass = loopmass
        for il in range(self.nloop):
            loopmass[il] = bodymass[Targets[il,0]]
            for ilb in range(loopnb[il]):
                ib = Targets[il,ilb]
                assert loopmass[il] == bodymass[ib]

        loopcharge = np.zeros((self.nloop), dtype=np.float64)
        self._loopcharge = loopcharge
        for il in range(self.nloop):
            loopcharge[il] = bodycharge[Targets[il,0]]
            for ilb in range(loopnb[il]):
                ib = Targets[il,ilb]
                assert loopcharge[il] == bodycharge[ib]

        self.BodyGraph = BodyGraph

    @cython.final
    def BuildSegmGraph(self):

        cdef Py_ssize_t ib, iint
        cdef Py_ssize_t ibp, iintp
        cdef Py_ssize_t isegm

        for Sym in self.Sym_list:
            if (Sym.TimeRev == -1):
                self.nint_min *= 2
                break

        # Making sure nint_min is big enough
        self.SegmGraph, self.nint_min = Build_SegmGraph_NoPb(self.nbody, self.nint_min, self.Sym_list)

        # Making sure ib -> self._bodysegm[ib, 0] is increasing
        isegm = 0
        self._bodysegm = -np.ones((self.nbody, self.nint_min), dtype = np.intp)
        for iint in range(self.nint_min):
            for ib in range(self.nbody):
                if self._bodysegm[ib, iint] < 0:

                    self._bodysegm[ib, iint] = isegm

                    segm_source = (ib, iint)
                    for edge in networkx.dfs_edges(self.SegmGraph, source=segm_source):
                        ibp = edge[1][0]
                        iintp = edge[1][1]
                        self._bodysegm[ibp, iintp] = isegm

                    isegm += 1

        self.nsegm = isegm

    @cython.final
    @cython.cdivision(True)
    def ChooseLoopGen(self):

        cdef Py_ssize_t il, ilb, isegm, ib, iint, ishift, n_nnid, jint
        cdef Py_ssize_t n_nnid_min, unique_size

        cdef ActionSym Sym
    
        self._loopgen = np.empty((self.nloop), dtype = np.intp)
        self._ngensegm_loop = np.empty((self.nloop), dtype = np.intp)
        self._gensegm_loop_start = np.empty((self.nloop), dtype = np.intp)
        self._gensegm_to_body = np.empty((self.nsegm), dtype = np.intp)
        self._gensegm_to_iint = np.empty((self.nsegm), dtype = np.intp)
        self._gensegm_to_iintrel = np.empty((self.nsegm), dtype = np.intp)

        cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] shifted_bodysegm = np.zeros((self.nint_min), dtype = np.intp)

        for il in range(self.nloop):

            n_nnid_min = self.nsegm + 1

            ib = self._Targets[il,0]
            for ishift in range(self.nint_min):

                for iint in range(self.nint_min):
                    jint = (iint + ishift) % self.nint_min
                    shifted_bodysegm[iint] = self._bodysegm[ib, jint]

                unique, unique_indices = np.unique(shifted_bodysegm, return_index = True)
                assert (unique == shifted_bodysegm[unique_indices]).all()

                unique_size = unique.size

                BodyHasContiguousGeneratingSegments = ((unique_indices.max()+1) == unique_size)

                # I need contiguous segments
                if BodyHasContiguousGeneratingSegments:
                    
                    n_nnid = 0
                    for iint in range(unique_size):
                        jint = (iint + ishift) % self.nint_min
                        Sym = self.intersegm_to_all[ib][jint]
                        if not(Sym.IsIdentityRot()):
                            n_nnid += 1

                    # I want to minimize the number of non identity
                    if n_nnid_min > n_nnid:

                        n_nnid_min = n_nnid

                        self._loopgen[il] = ib
                        self._ngensegm_loop[il] = unique_size
                        self._gensegm_loop_start[il] = ishift

                        for iint in range(self._ngensegm_loop[il]):
                            jint = (iint + ishift) % self.nint_min
                            isegm = self._bodysegm[ib, jint]

                            self._gensegm_to_body[isegm] = ib
                            self._gensegm_to_iint[isegm] = jint
                            self._gensegm_to_iintrel[isegm] = iint

    @cython.final
    def ChooseInterSegm(self):

        cdef Py_ssize_t iint, ib, isegm

        # Choose interacting segments as earliest possible times.
        intersegm_to_body = np.zeros((self.nsegm), dtype = np.intp)
        intersegm_to_iint = np.zeros((self.nsegm), dtype = np.intp)

        self._intersegm_to_body = intersegm_to_body
        self._intersegm_to_iint = intersegm_to_iint

        assigned_segms = set()

        for iint in range(self.nint_min):
            for ib in range(self.nbody):

                isegm = self._bodysegm[ib,iint]

                if not(isegm in assigned_segms):
                    
                    intersegm_to_body[isegm] = ib
                    intersegm_to_iint[isegm] = iint
                    assigned_segms.add(isegm)

        #Every interaction happens at iint == 0
        if not (intersegm_to_iint == 0).all():
            raise ValueError("Catastrophic failure: interacting segments are not all at initial positions")

    @cython.final
    def GatherInterSym(self):

        cdef Py_ssize_t isegm
        cdef ActionSym Sym
        
        InterTimeRev = np.zeros((self.nsegm), dtype=np.intp)
        InterSpaceRotPos = np.zeros((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        InterSpaceRotPosIsId = np.zeros((self.nsegm), dtype=np.intc)
        InterSpaceRotVel = np.zeros((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        InterSpaceRotVelIsId = np.zeros((self.nsegm), dtype=np.intc)

        self._InterTimeRev = InterTimeRev
        self._InterSpaceRotPos = InterSpaceRotPos
        self._InterSpaceRotPosIsId = InterSpaceRotPosIsId
        self._InterSpaceRotVel = InterSpaceRotVel
        self._InterSpaceRotVelIsId = InterSpaceRotVelIsId

        for isegm in range(self.nsegm):

            ib = self._gensegm_to_body[isegm]
            iint = self._gensegm_to_iint[isegm]

            Sym = self.intersegm_to_all[ib][iint]
            
            InterTimeRev[isegm] = Sym.TimeRev
            InterSpaceRotPos[isegm,:,:] = Sym.SpaceRot
            self._InterSpaceRotPosIsId[isegm] = Sym.IsIdentityRot()

            Sym = Sym.TimeDerivative()
            InterSpaceRotVel[isegm,:,:] = Sym.SpaceRot
            self._InterSpaceRotVelIsId[isegm] = Sym.IsIdentityRot()

    @cython.final
    def DetectSegmRequiresDisp(self):

        cdef Py_ssize_t ib , iint
        cdef Py_ssize_t ibp, iintp

        self._SegmRequiresDisp = - np.ones((self.nbody, self.nint_min), dtype=np.intc)
                    
        cdef ActionSym Identity, Sym, EdgeSym, NewSym, Plotted_Sym
                
        Identity = ActionSym.Identity(self.nbody, self.geodim)

        # More convoluted code because I want to make sure a path has a single color in the GUI
        for ib in range(self.nbody):

            for iint in range(self.nint_min):
        
                if self._SegmRequiresDisp[ib,iint] < 0:

                    self._SegmRequiresDisp[ib,iint] = 1

                    segm_source = (ib, iint)
                    Sym_dict = {segm_source : Identity}
                    Plotted_Syms = [Identity]
                    
                    for edge in networkx.dfs_edges(self.SegmGraph, source=segm_source):

                        ibp = edge[1][0]
                        iintp = edge[1][1]
                        
                        Sym = Sym_dict[edge[0]]
                        EdgeSym = self.SegmGraph.edges[edge]["SymList"][0]    

                        if edge[0] <= edge[1]:
                            NewSym = EdgeSym.Compose(Sym)
                        else:
                            NewSym = (EdgeSym.Inverse()).Compose(Sym)

                        Sym_dict[edge[1]] = NewSym

                        if ib == ibp:

                            for Plotted_Sym in Plotted_Syms:
                                if Plotted_Sym.IsSameRot(NewSym):
                                    self._SegmRequiresDisp[ibp,iintp] = 0
                                    break
                            else:
                                self._SegmRequiresDisp[ibp,iintp] = 1
                                Plotted_Syms.append(NewSym)

                    for edge in networkx.dfs_edges(self.SegmGraph, source=segm_source):

                        ibp = edge[1][0]

                        if ib != ibp:
                            iintp = edge[1][1]

                            NewSym = Sym_dict[edge[1]]

                            for Plotted_Sym in Plotted_Syms:
                                if Plotted_Sym.IsSameRot(NewSym):
                                    self._SegmRequiresDisp[ibp,iintp] = 0
                                    break
                            else:
                                self._SegmRequiresDisp[ibp,iintp] = 1
                                Plotted_Syms.append(NewSym)

        assert (np.asarray(self._SegmRequiresDisp) >= 0).all()

    @cython.final
    def Compute_n_sub_fft(self):
        
        self._n_sub_fft = np.zeros((self.nloop), dtype=np.intp)
        cdef Py_ssize_t il
        for il in range(self.nloop):

            assert  self.nint_min % self._ncoeff_min_loop[il] == 0
            assert (self.nint_min // self._ncoeff_min_loop[il]) % self._ngensegm_loop[il] == 0        
            
            self._n_sub_fft[il] = (self.nint_min // (self._ncoeff_min_loop[il] * self._ngensegm_loop[il]))

            if (self._n_sub_fft[il]) not in [1,2]:
                raise ValueError(f"Catastrophic failure : {self._n_sub_fft[il] = } not in [1,2]")

        cdef ActionSym Sym
        cdef Py_ssize_t isegm

        self._ALG_Iint = np.empty((self.nloop), dtype=np.intp)
        self._ALG_TimeRev = np.empty((self.nloop), dtype=np.intp)
        self._ALG_SpaceRotPos = np.empty((self.nloop, self.geodim, self.geodim), dtype=np.float64)
        self._ALG_SpaceRotVel = np.empty((self.nloop, self.geodim, self.geodim), dtype=np.float64)

        cdef Py_ssize_t iint_uneven, ib
        cdef Py_ssize_t idim, jdim
        for il in range(self.nloop):
            
            iint_uneven = self._ngensegm_loop[il] % self.nint_min

            ib = self._loopgen[il]
            isegm = self._bodysegm[ib, iint_uneven]
            Sym = self.gensegm_to_all[ib][iint_uneven]

            self._ALG_Iint[il] = self._gensegm_to_iint[isegm]
            self._ALG_TimeRev[il] = Sym.TimeRev
            self._ALG_SpaceRotPos[il,:,:] = Sym._SpaceRot[:,:]

            Sym = Sym.TimeDerivative()
            self._ALG_SpaceRotVel[il,:,:] = Sym._SpaceRot[:,:]

        self._PerDefBeg_Isegm = np.empty((self.nsegm), dtype=np.intp)
        self._PerDefBeg_TimeRev = np.empty((self.nsegm), dtype=np.intp)
        self._PerDefBeg_SpaceRotPos = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        self._PerDefBeg_SpaceRotVel = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)

        for isegm in range(self.nsegm):

            iint_uneven = self.nint_min - 1

            ib = self._intersegm_to_body[isegm]

            self._PerDefBeg_Isegm[isegm] = self._bodysegm[ib, iint_uneven]

            Sym = self.intersegm_to_all[ib][iint_uneven]
            self._PerDefBeg_TimeRev[isegm] = Sym.TimeRev
            self._PerDefBeg_SpaceRotPos[isegm,:,:] = Sym._SpaceRot[:,:]

            Sym = Sym.TimeDerivative()
            self._PerDefBeg_SpaceRotVel[isegm,:,:] = Sym._SpaceRot[:,:]

        self._PerDefEnd_Isegm = np.empty((self.nsegm), dtype=np.intp)
        self._PerDefEnd_TimeRev = np.empty((self.nsegm), dtype=np.intp)
        self._PerDefEnd_SpaceRotPos = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        self._PerDefEnd_SpaceRotVel = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)

        for isegm in range(self.nsegm):

            iint_uneven = 1 % self.nint_min

            ib = self._intersegm_to_body[isegm]

            self._PerDefEnd_Isegm[isegm] = self._bodysegm[ib, iint_uneven]

            Sym = self.intersegm_to_all[ib][iint_uneven]
            self._PerDefEnd_TimeRev[isegm] = Sym.TimeRev
            self._PerDefEnd_SpaceRotPos[isegm,:,:] = Sym._SpaceRot[:,:]

            Sym = Sym.TimeDerivative()
            self._PerDefEnd_SpaceRotVel[isegm,:,:] = Sym._SpaceRot[:,:]

    @cython.final
    def ConfigureShortcutSym(self, double eps=1e-14):

        cdef Py_ssize_t il

        self._ParamBasisShortcutPos_th = np.empty((self.nloop), dtype=np.intc)
        self._ParamBasisShortcutVel_th = np.empty((self.nloop), dtype=np.intc)

        for il in range(self.nloop):

            self._ParamBasisShortcutPos_th[il] = GENERAL_SYM
            self._ParamBasisShortcutVel_th[il] = GENERAL_SYM

            if self._nnz_k_shapes[il,0] == 1:
                if self._nnz_k_buf[self._nnz_k_shifts[il]] == 0:

                    if (2*self._params_basis_shapes[il,0]) == self._params_basis_shapes[il,2]:

                        params_basis = self.params_basis_pos(il)
                        m = params_basis.shape[0]
                        n = params_basis.shape[2]
                        
                        params_basis_r = np.empty((m, 2, n),dtype=np.float64)
                        params_basis_r[:,0,:] = params_basis[:,0,:].real
                        params_basis_r[:,1,:] = params_basis[:,0,:].imag

                        if np.linalg.norm(params_basis_r.reshape(n,n) - np.identity(params_basis.shape[2])) < eps:

                            self._ParamBasisShortcutPos_th[il] = RFFT

                        params_basis = self.params_basis_vel(il)
                        m = params_basis.shape[0]
                        n = params_basis.shape[2]
                        
                        params_basis_r = np.empty((m, 2, n),dtype=np.float64)
                        params_basis_r[:,0,:] = params_basis[:,0,:].real
                        params_basis_r[:,1,:] = params_basis[:,0,:].imag

                        if np.linalg.norm(params_basis_r.reshape(n,n) - np.identity(params_basis.shape[2])) < eps:

                            self._ParamBasisShortcutVel_th[il] = RFFT

    @cython.final
    def AssertAllSegmGenConstraintsAreRespected(self, all_pos, eps=1e-12, pos=True, verbose=False):

        for il in range(self.nloop):

            ib = self._loopgen[il] # because only loops have been computed in all_pos so far.
            
            for iint in range(self.nint_min):

                isegm = self._bodysegm[ib, iint]
                
                if pos:
                    Sym = self.gensegm_to_all[ib][iint]
                else:
                    Sym = self.gensegm_to_all[ib][iint].TimeDerivative()
                
                ib_source = self._gensegm_to_body[isegm]
                iint_source = self._gensegm_to_iint[isegm]
                il_source = self._bodyloop[ib_source]
                assert il_source == il

                ibeg_source = iint_source * self.segm_size          
                iend_source = ibeg_source + self.segm_size
                assert iend_source <= self._nint

                # One position at a time
                for iint_s in range(ibeg_source, iend_source):
                    
                    tnum, tden = Sym.ApplyT(iint_s, self._nint)

                    assert self._nint % tden == 0
                    iint_t = (tnum * (self._nint // tden) + self._nint) % self._nint

                    xs = np.matmul(Sym.SpaceRot, all_pos[il, iint_s,:])
                    xt = all_pos[il, iint_t,:]
                    dx = xs - xt

                    assert (np.linalg.norm(dx)) < eps
                
                # All positions at once
                tnum_target, tden_target = Sym.ApplyTSegm(iint_source, self.nint_min)
                assert self.nint_min % tden_target == 0
                iint_target = (tnum_target * (self.nint_min // tden_target) + self.nint_min) % self.nint_min   
                    
                ibeg_target = iint_target * self.segm_size         
                iend_target = ibeg_target + self.segm_size
                
                # IMPORTANT !!!!
                if Sym.TimeRev == -1:
                    ibeg_target += 1
                    iend_target += 1
                
                pos_source_segm = all_pos[il,ibeg_source:iend_source,:]
                pos_target_segm = np.empty((self.segm_size, self.geodim), dtype=np.float64)

                Sym.TransformSegment(pos_source_segm, pos_target_segm)

                if iend_target <= self._nint:
                    err = (np.linalg.norm(pos_target_segm - all_pos[il, ibeg_target:iend_target, :]))
                    if verbose:
                        print(err)
                    assert err < eps
                else:

                    err = (np.linalg.norm(pos_target_segm[:self.segm_size-1,:] - all_pos[il, ibeg_target:iend_target-1, :]))
                    if verbose:
                        print(err)
                    assert iend_target == self._nint+1
                    assert err < eps
                    assert (np.linalg.norm(pos_target_segm[ self.segm_size-1,:] - all_pos[il, 0, :])) < eps
            
    @cython.final
    def AssertAllBodyConstraintAreRespected(self, all_pos, eps=1e-12, pos=False, verbose=False):
        # Make sure loop constraints are respected
        
        for il, Constraints in enumerate(self.LoopGenConstraints):

            for icstr, Sym in enumerate(Constraints):

                if not pos:
                    Sym = Sym.TimeDerivative()

                assert (self._nint % Sym.TimeShiftDen) == 0

                ConstraintIsRespected = True

                for iint in range(self._nint):

                    tnum, tden = Sym.ApplyT(iint, self._nint)
                    jint = tnum * self._nint // tden
                    
                    err = np.linalg.norm(all_pos[il,jint,:] - np.matmul(Sym.SpaceRot, all_pos[il,iint,:]))
                    if verbose:
                        print(err)

                    assert (err < eps)

    @cython.final
    @cython.cdivision(True)
    def Set_inter_law(self, inter_law = None, inter_law_str = None, inter_law_param_dict = None):
        """ Sets the interaction law of the system.

        There are several ways to set the interaction law:

        * Through **inter_law**. This argument can be:
            * A C function with signature ``void (double, double *, void *)``.
                * The first argument denotes the squared inter-body distance : :math:`\Delta x^2 = \sum_{i=0}^{\\text{geodim}-1} \Delta x_i^2`
                * The second argument denotes an array of :obj:`numpy:numpy.float64` with 3 elements. The function should write the value of the interaction, its first, and its second derivative to the first, second, and third elements of this array respectively. **WARNING** : These derivatives should be understood with respect to the variable :math:`\Delta x^2`, namely the **squared** inter-body distance.
                * The third argument denotes a pointer to parameters that can be used during the computation.
            * A `Python <https://www.python.org/>`_ function that will be compiled to a C function using `Numba <https://numba.pydata.org/>`_ if available on the user's system. For performance reasons, using pure Python functions is not allowed. The arguments are similar to the ones defined above in the case of a C function. For instance, the following function defines the Newtonian gravitational potential:

        .. code-block:: Python

            def inter_law(xsq, res, ptr):

                a = xsq ** (-2.5)
                b = xsq*a
            
                res[0] = -xsq*b
                res[1]= 0.5*b
                res[2] = (-0.75)*a

        * Through **inter_law_str**, whose possible values are:
            * ``"gravity_pot"`` : the potential is the classical Newtonian gravitational potential: :math:`V(x) = \\frac{1}{x}`
            * ``"power_law_pot"`` : the potential is a power law : :math:`V(x) = x^n`. Its parameters should be given through the :class:`python:dict` **inter_law_param_dict** with key "n".
            * A string defining a `Python <https://www.python.org/>`_ function to be compiled with `Numba <https://numba.pydata.org/>`_ . The process is similar to passing a `Python <https://www.python.org/>`_ function directly as described above.

        * If both **inter_law** and **inter_law_str** are :data:`python:None`, the interaction law is the classical gravitational potential.

        Parameters
        ----------
        inter_law : optional
            Function defining the interaction law, by default :data:`python:None`.
        inter_law_str : :class:`python:str`, optional
            Description of the interaction law dictating the dynamics of the system, by default :data:`python:None`.
        inter_law_param_dict : :class:`python:dict`, optional
            Parameters pertaining to the interaction law, by default :data:`python:None`.

        """    

        cdef ccallback_t callback_inter_fun

        if inter_law_str is None:
            if inter_law is None:
                self._inter_law = gravity_pot
                self.inter_law_str = "gravity_pot"
                self.inter_law_param_dict = None
                self._inter_law_param_ptr = NULL
            else:

                if inter_law_param_dict is not None:
                    raise ValueError('Argument inter_law_param_dict should be None.')

                ccallback_prepare(&callback_inter_fun, signatures, inter_law, CCALLBACK_DEFAULTS)

                if (callback_inter_fun.py_function != NULL):

                    if not NUMBA_AVAILABLE:
                        raise ValueError("Numba is not available and provided inter_law is a Python function. Using a Python function is disallowed for performance reasons. Please provide a C function or install Numba on your system.")

                    inter_law_numba = jit_inter_law(inter_law)
                    ccallback_prepare(&callback_inter_fun, signatures, inter_law_numba, CCALLBACK_DEFAULTS)
                    
                    if (callback_inter_fun.signature.value != 0):
                        raise ValueError(f"Provided inter_law is a Python function with incorrect signature.")

                    self.inter_law_str = inspect.getsource(inter_law)

                elif (callback_inter_fun.signature.value != 0):
                    raise ValueError(f"Provided inter_law is a C function with incorrect signature. Signature should be {signatures[0].signature}.")
                else:

                    self.inter_law_str = "Custom C function"

                self._inter_law = <inter_law_fun_type> callback_inter_fun.c_function

        else:
            if inter_law is not None:
                raise ValueError("Please provide either inter_law or inter_law_str, not both.")
        
            if not isinstance(inter_law_str, str):
                raise ValueError(f"Provided inter_law_str is of type {type(inter_law_str)} but should be of type string.")

            if inter_law_str == "gravity_pot":
                self._inter_law = gravity_pot
                self.inter_law_str = "gravity_pot"
                self.inter_law_param_dict = None
                self._inter_law_param_ptr = NULL
            
            elif inter_law_str.startswith("power_law_pot"):

                self._inter_law = power_law_pot
                self.inter_law_param_dict = { key : inter_law_param_dict[key] for key in ("n")}

                n = self.inter_law_param_dict["n"] / 2 # argument is xsq !
                nm2 = n-2
                mnnm1 = -n*(n-1)
                
                self.inter_law_str = f"power_law_pot({n=})"
                self._inter_law_param_buf = np.array([-n, mnnm1, nm2], dtype=np.float64)
                self._inter_law_param_ptr = <void*> &self._inter_law_param_buf[0]
            else:

                if not NUMBA_AVAILABLE:
                    raise ValueError("Numba is not available and provided inter_law_str is a string defining a Python function. Using a Python function is disallowed for performance reasons. Please provide a C function or install Numba on your system.")

                if inter_law_param_dict is not None:
                    raise ValueError('Argument inter_law_param_dict should be None.')

                try:
                    
                    inter_law_numba = jit_inter_law_str(inter_law_str)
                    ccallback_prepare(&callback_inter_fun, signatures, inter_law_numba, CCALLBACK_DEFAULTS)

                    if (callback_inter_fun.signature.value != 0):
                        raise ValueError(f"Provided inter_law_str defines a Python function whose corresponding C function has incorrect signature. Signature should be {signatures[0].signature}.")
                    
                    self.inter_law_str = inter_law_str
                    self.inter_law_param_dict = None
                    self._inter_law = <inter_law_fun_type> callback_inter_fun.c_function
                    self._inter_law_param_ptr = NULL

                except Exception as err:
                    print(err)
                    raise ValueError("Could not compile provided string.")

        if not(self.Validate_inter_law()):
            raise ValueError(f'Finite differences could not validate the provided potential law.')

        self.LawIsHomo, self.Homo_exp, self.Homo_unit = self.Detect_homo_inter_law()

    @cython.final
    @cython.cdivision(True)
    def Validate_inter_law(self, double xsqo=1., double dxsq=1e-4,  double eps=1e-7, bint verbose=False):

        cdef double xsqp = xsqo + dxsq
        cdef double xsqm = xsqo - dxsq

        cdef double[3] poto
        cdef double[3] potp
        cdef double[3] potm

        self._inter_law(xsqo, poto, self._inter_law_param_ptr)
        self._inter_law(xsqp, potp, self._inter_law_param_ptr)
        self._inter_law(xsqm, potm, self._inter_law_param_ptr)

        cdef double fd1 = (potp[0] - potm[0]) / (2*dxsq)
        cdef double fd2 = ((potp[0] - poto[0]) + (potm[0] - poto[0])) / (dxsq*dxsq)
        
        cdef double err_fd1 = cfabs(fd1 - poto[1])
        cdef double err_fd2 = cfabs(fd2 - poto[2])

        if verbose:
            print(err_fd1)
            print(err_fd2)

        cdef bint FD1_OK = (err_fd1 < eps)
        cdef bint FD2_OK = (err_fd2 < eps)

        return (FD1_OK and FD2_OK)

    @cython.final
    @cython.cdivision(True)
    def Detect_homo_inter_law(self, double xsqo=1., double fac=1.1, Py_ssize_t n=10, double eps=1e-10):

        cdef double[3] pot
        cdef double xsq = xsqo

        cdef Py_ssize_t i
        cdef double[::1] alpha_approx = np.empty((n), dtype=np.float64)
        cdef double alpha_avg = 0

        for i in range(n):

            self._inter_law(xsq, pot, self._inter_law_param_ptr)

            alpha_approx[i] = xsq * pot[1] / pot[0]
            alpha_avg += alpha_approx[i]

            xsq *= fac    

        alpha_avg /= n

        cdef bint IsHomo = True

        for i in range(n):
            IsHomo = IsHomo and cfabs(alpha_approx[i] - alpha_avg) < eps
        
        self._inter_law(1., pot, self._inter_law_param_ptr)

        return IsHomo, alpha_avg, pot[0]

    # Should really be a class method
    @cython.final
    cpdef all_coeffs_pos_to_vel_inplace(
        self                                ,
        double complex[:,:,::1] all_coeffs  ,
    ):

        cdef int nloop = all_coeffs.shape[0]
        cdef int ncoeffs = all_coeffs.shape[1]
        cdef int geodim = all_coeffs.shape[2]

        cdef Py_ssize_t il, k, idim

        cdef double complex fac = 1j*ctwopi

        for il in range(nloop):
            for k in range(ncoeffs):
                for idim in range(geodim):
                    # all_coeffs[il,k,idim] *= fac*k # Causes weird Cython error on Windows
                    all_coeffs[il,k,idim] = all_coeffs[il,k,idim] * (fac*k)

    @cython.final
    def Get_segmpos_minmax(
        self                    ,
        double[:,:,::1] segmpos ,
    ):

        segmpos_minmax_np = np.empty((self.nsegm,2,self.geodim), dtype=np.float64)
        cdef double[:,:,::1] segmpos_minmax = segmpos_minmax_np

        cdef Py_ssize_t iint, idim, isegm
        cdef double* mi = <double*> malloc(sizeof(double)*self.geodim)
        cdef double* ma = <double*> malloc(sizeof(double)*self.geodim)

        cdef double* segmpos_ptr = &segmpos[0,0,0]
        cdef double val

        for isegm in range(self.nsegm):

            for idim in range(self.geodim):

                mi[idim] =  DBL_MAX
                ma[idim] = -DBL_MAX

            for iint in range(self.segm_store):
                for idim in range(self.geodim):

                    val = segmpos_ptr[0]

                    mi[idim] = min(mi[idim], val)
                    ma[idim] = max(ma[idim], val)

                    segmpos_ptr += 1
        
            for idim in range(self.geodim):

                segmpos_minmax[isegm,0,idim] = mi[idim]
                segmpos_minmax[isegm,1,idim] = ma[idim]

        free(mi)
        free(ma)

        return segmpos_minmax_np

    @cython.final
    def GetFullAABB(
        self                    ,
        double[:,:,::1] segmpos ,
        double extend=0.        ,
        bint MakeSquare = 0     ,
    ):

        cdef double[:,:,::1] segmpos_minmax = self.Get_segmpos_minmax(segmpos)

        cdef double[:,::1] RotMat

        AABB_np = np.empty((2,self.geodim), dtype=np.float64)
        cdef double[:,::1] AABB = AABB_np
        AABB[:,:] = segmpos_minmax[0,:,:]

        cdef Py_ssize_t iint, ib, isegm
        cdef Py_ssize_t idim, jdim
        cdef double x, dx

        cdef Py_ssize_t[::1] iminmax = np.empty((self.geodim), dtype=np.intp)

        # Can't iterate on a memoryview directly
        for tuple_it in itertools.product(range(2), repeat=self.geodim):

            for idim in range(self.geodim):
                iminmax[idim] = tuple_it[idim]

            for ib in range(self.nbody):
                for iint in range(self.nint_min):

                    isegm = self._bodysegm[ib,iint]
                    RotMat = self.gensegm_to_all[ib][iint].SpaceRot

                    for idim in range(self.geodim):

                        x = 0
                        for jdim in range(self.geodim):
                            x += RotMat[idim,jdim]*segmpos_minmax[isegm, iminmax[jdim], jdim]

                        AABB[0,idim] = min(AABB[0,idim], x)
                        AABB[1,idim] = max(AABB[1,idim], x)

        for i in range(self.geodim):
            dx = extend * (AABB[1,idim] - AABB[0,idim])
            AABB[0,idim] -= dx
            AABB[1,idim] += dx

        if MakeSquare:

            dx = 0.
            for idim in range(self.geodim):
                dx = max(dx, AABB[1,idim] - AABB[0,idim])
            dx = 0.5 * dx

            for idim in range(self.geodim):
                x = 0.5 * (AABB[1,idim] + AABB[0,idim])
                AABB[0,idim] = x - dx
                AABB[1,idim] = x + dx

        return AABB_np

    @cython.final
    def Init_to_dict(self):

        bodymass = [self._loopmass[self._bodyloop[ib]] for ib in range(self.nbody)]
        bodycharge = [self._loopcharge[self._bodyloop[ib]] for ib in range(self.nbody)]

        Info_dict = {
            "choreo_version" : choreo.metadata.__version__          ,
            "geodim" : self.geodim                                  ,
            "nbody" : self.nbody                                    ,
            "bodymass" : bodymass                                   ,
            "bodycharge" : bodycharge                               ,
            "Sym_list" : [Sym.to_dict() for Sym in self.Sym_list]   ,
            "inter_law_str" : self.inter_law_str                    ,
        }

        if self.inter_law_param_dict is not None:
            Info_dict["inter_law_param_dict"] = self.inter_law_param_dict

        return Info_dict

    @cython.final
    def Segmpos_Descriptor(self,
        double[::1] params_mom_buf = None   ,
        segmpos = None  , segmvel = None    , 
        Action = None   , Gradaction = None , Hash_Action = None,
        extend = 0.03   ,
    ):

        if segmpos is None:
            if params_mom_buf is None:
                raise ValueError("Missing params_mom_buf.")
            segmpos = self.params_to_segmpos(params_mom_buf)

        if segmvel is None:
            if params_mom_buf is None:
                raise ValueError("Missing params_mom_buf.")
            segmvel = self.params_to_segmvel(params_mom_buf)

        if Action is None:
            if params_mom_buf is None:
                raise ValueError("Missing params_mom_buf.")
            Action = self.segmpos_params_to_action(segmpos, params_mom_buf)

        if Gradaction is None:
            if params_mom_buf is None:
                raise ValueError("Missing params_mom_buf.")
            Gradaction_vect = self.segmpos_params_to_action_grad(segmpos, params_mom_buf)
            Gradaction = np.linalg.norm(Gradaction_vect)

        if Hash_Action is None:
            Hash_Action = self.segmpos_to_hash(segmpos)

        loop_len, bin_dx_min = self.segm_to_path_stats(segmpos, segmvel)
        AABB = self.GetFullAABB(segmpos, extend)

        Info_dict = {}

        Info_dict["nint_min"] = self.nint_min
        Info_dict["nint"] = self._nint
        Info_dict["segm_size"] = self.segm_size
        Info_dict["segm_store"] = self.segm_store

        Info_dict["nloop"] = self.nloop
        Info_dict["loopnb"] = self.loopnb.tolist()
        Info_dict["Targets"] = self.Targets.tolist() 

        Info_dict["Action"] = Action
        Info_dict["Grad_Action"] = Gradaction
        Info_dict["Min_Bin_Distance"] = bin_dx_min.tolist()
        Info_dict["Loop_Length"] = loop_len.tolist()
        Info_dict["Max_PathLength"] = loop_len.max()

        Info_dict["Hash"] = Hash_Action.tolist()
        Info_dict["AABB"] = AABB.tolist()

        Info_dict["nsegm"] = self.nsegm
        Info_dict["bodysegm"] = self.bodysegm.tolist()
        Info_dict["SegmRequiresDisp"] = self.SegmRequiresDisp.tolist()

        InterSegmSpaceRot = []
        InterSegmTimeRev = []

        for ib in range(self.nbody):
            InterSegmSpaceRot_b = []
            InterSegmTimeRev_b = []
            for iint in range(self.nint_min):

                Sym = self.intersegm_to_all[ib][iint]

                InterSegmSpaceRot_b.append(Sym.SpaceRot.tolist())
                InterSegmTimeRev_b.append(Sym.TimeRev)

            InterSegmSpaceRot.append(InterSegmSpaceRot_b)
            InterSegmTimeRev.append(InterSegmTimeRev_b)
        
        Info_dict["InterSegmSpaceRot"] = InterSegmSpaceRot
        Info_dict["InterSegmTimeRev"] = InterSegmTimeRev

        return Info_dict

    @cython.final
    def Write_Descriptor(self, filename = None, **kwargs):

        Info_dict = self.Init_to_dict()

        Info_dict.update(self.Segmpos_Descriptor(**kwargs))

        with open(filename, "w") as jsonFile:
            jsonString = json.dumps(Info_dict, indent=4, sort_keys=False)
            jsonFile.write(jsonString)

    @cython.final
    @staticmethod
    def FromDict(InfoDict):
    
        geodim = InfoDict["geodim"]
        nbody = InfoDict["nbody"]
        bodymass = np.array(InfoDict["bodymass"], dtype=np.float64)
        bodycharge = np.array(InfoDict["bodycharge"], dtype=np.float64)
        Sym_list = [ActionSym.FromDict(Sym_dict) for Sym_dict in InfoDict["Sym_list"]]
        
        inter_law_str = InfoDict["inter_law_str"]
        inter_law_param_dict = InfoDict.get("inter_law_param_dict")

        return NBodySyst(
            geodim, nbody, bodymass, bodycharge, Sym_list,
            inter_law_str = inter_law_str, inter_law_param_dict = inter_law_param_dict
        )

    @cython.final
    @staticmethod
    def FromSolutionFile(file_basename):
    
        with open(file_basename+'.json') as jsonFile:
            InfoDict = json.load(jsonFile)

        NBS = NBodySyst.FromDict(InfoDict)
        NBS.nint = InfoDict["nint"]

        assert NBS.segm_size == InfoDict["segm_size"]

        if InfoDict["segm_size"] != InfoDict["segm_store"]:
            NBS.ForceGreaterNStore = True
            assert NBS.segm_store == InfoDict["segm_store"]

        assert NBS.nsegm == InfoDict["nsegm"]

        segmpos = np.load(file_basename+'.npy')
        assert segmpos.shape[0] == NBS.nsegm
        assert segmpos.shape[1] == NBS.segm_store
        assert segmpos.shape[2] == NBS.geodim

        return NBS, segmpos

    @cython.final
    def GetKrylovJacobian(self, Use_exact_Jacobian=True, jac_options_kw={}):

        if (Use_exact_Jacobian):

            jacobian = scipy.optimize.nonlin.KrylovJacobian(**jac_options_kw)

            def matvec(self,v):                
                return self.NBS.segmpos_dparams_to_action_hess(self.segmpos, v)

            def update(self, x, f):
                self.segmpos = self.NBS.params_to_segmpos(x)
                scipy.optimize.nonlin.KrylovJacobian.update(self, x, f)

            def setup(self, x, f, func):
                self.segmpos = self.NBS.params_to_segmpos(x)
                scipy.optimize.nonlin.KrylovJacobian.setup(self, x, f, func)

            jacobian.matvec = types.MethodType(matvec, jacobian)
            jacobian.rmatvec = types.MethodType(matvec, jacobian)
            jacobian.update = types.MethodType(update, jacobian)
            jacobian.setup = types.MethodType(setup, jacobian)

        else: 

            jacobian = scipy.optimize.nonlin.KrylovJacobian(**jac_options_kw)
        
        jacobian.NBS = self

        return jacobian

    @cython.final
    @cython.cdivision(True)
    def TestHashSame(self, double[::1] Hash_a, double[::1] Hash_b, double rtol=1e-5, bint detect_multiples=True):

        cdef Py_ssize_t nhash = self._Hash_exp.shape[0]

        assert Hash_a.shape[0] == nhash
        assert Hash_b.shape[0] == nhash
        
        cdef double[::1] all_test = np.zeros(self._Hash_exp.shape[0])
        cdef Py_ssize_t ihash
        cdef double pow_fac_m, refval

        if detect_multiples and self.LawIsHomo:

            pow_fac_m = (self.Homo_exp-1)/(2*self._Hash_exp[0])

            refval = cpow(Hash_a[0] / Hash_b[0], pow_fac_m)

            for ihash in range(1,nhash):

                pow_fac_m = (self.Homo_exp-1)/(2*self._Hash_exp[ihash])

                all_test[ihash] = cpow(Hash_a[ihash]/Hash_b[ihash], pow_fac_m) - refval

        else:

            for ihash in range(nhash):

                all_test[ihash] = (Hash_b[ihash]-Hash_a[ihash]) / (Hash_b[ihash] + Hash_a[ihash])

        IsSame = (np.linalg.norm(all_test, np.inf) < rtol)

        return IsSame

    @cython.final
    @cython.cdivision(True)
    def TestActionSame(self, double Action_a, double Action_b, double rtol=1e-5):

        return cfabs(Action_a - Action_b) / (cfabs(Action_a) + cfabs(Action_b)) < rtol

    @cython.final
    def DetectXlim(self, segmpos):

        assert segmpos.shape[1] == self.segm_store

        xmin_segm_np = segmpos.min(axis=1)
        xmax_segm_np = segmpos.max(axis=1)

        cdef double[:,::1] xmin_segm = xmin_segm_np
        cdef double[:,::1] xmax_segm = xmax_segm_np

        xmin_np = xmin_segm_np.min(axis=0)
        xmax_np = xmax_segm_np.min(axis=0)

        cdef double[::1] xmin = xmin_np
        cdef double[::1] xmax = xmax_np

        cdef double[::1] x

        cdef Py_ssize_t ib, iint, idim
        cdef Py_ssize_t isegm

        for ib in range(self.nbody):
            for iint in range(self.nint_min):

                if self._SegmRequiresDisp[ib,iint]:

                    Sym = self.intersegm_to_all[ib][iint]
                    isegm = self._bodysegm[ib, iint]

                    x = np.matmul(Sym.SpaceRot, xmin_segm[isegm,:])

                    for idim in range(self.geodim):
                        if x[idim] < xmin[idim]:
                            xmin[idim] = x[idim]
                        if x[idim] > xmax[idim]:
                            xmax[idim] = x[idim]

                    x = np.matmul(Sym.SpaceRot, xmax_segm[isegm,:])

                    for idim in range(self.geodim):
                        if x[idim] < xmin[idim]:
                            xmin[idim] = x[idim]
                        if x[idim] > xmax[idim]:
                            xmax[idim] = x[idim]

        return np.asarray(xmin), np.asarray(xmax)

    @cython.final
    @cython.cdivision(True)
    def DetectEscape(self, segmpos, double fac = 2.):

        assert segmpos.shape[1] == self.segm_store 

        cdef double[:,::1] xmin_segm = segmpos.min(axis=1)
        cdef double[:,::1] xmax_segm = segmpos.max(axis=1)

        cdef double[:,::1] xmin_body = np.full((self.nbody, self.geodim),  np.inf, dtype=np.float64)
        cdef double[:,::1] xmax_body = np.full((self.nbody, self.geodim), -np.inf, dtype=np.float64)
        cdef double[:,::1] xmid_body = np.empty((self.nbody, self.geodim), dtype=np.float64)
        cdef double[::1] size_body = np.empty((self.nbody), dtype=np.float64)

        cdef double[::1] x

        cdef Py_ssize_t ib, ibp, iint, idim
        cdef Py_ssize_t isegm

        cdef double dist, size, dx

        for ib in range(self.nbody):
            for iint in range(self.nint_min):

                Sym = self.intersegm_to_all[ib][iint]
                isegm = self._bodysegm[ib, iint]

                x = np.matmul(Sym.SpaceRot, xmin_segm[isegm,:])

                for idim in range(self.geodim):
                    if x[idim] < xmin_body[ib,idim]:
                        xmin_body[ib,idim] = x[idim]
                    if x[idim] > xmax_body[ib,idim]:
                        xmax_body[ib,idim] = x[idim]

                x = np.matmul(Sym.SpaceRot, xmax_segm[isegm,:])

                for idim in range(self.geodim):
                    if x[idim] < xmin_body[ib,idim]:
                        xmin_body[ib,idim] = x[idim]
                    if x[idim] > xmax_body[ib,idim]:
                        xmax_body[ib,idim] = x[idim]

            for idim in range(self.geodim):
                xmid_body[ib, idim] = (xmax_body[ib,idim] + xmin_body[ib,idim]) / 2

            size = 0
            for idim in range(self.geodim):
                dx = xmax_body[ib,idim] - xmin_body[ib,idim] 
                size += dx*dx
            size_body[ib] = csqrt(size)

        BodyGraph = networkx.Graph()
        for ib in range(self.nbody):
            BodyGraph.add_node(ib)

        for ib in range(self.nbody):
            for ibp in range(ib+1,self.nbody):

                dist = 0
                for idim in range(self.geodim):
                    dx = xmid_body[ib,idim] - xmid_body[ibp,idim] 
                    dist += dx*dx
                dist = csqrt(dist)

                if (dist < fac * (size_body[ib] + size_body[ibp])):

                    BodyGraph.add_edge(ib,ibp)

        return not(networkx.is_connected(BodyGraph))

    @cython.final
    def plot_segmpos_2D(self, segmpos, filename, fig_size=(10,10), dpi=100, color=None, color_list=None, xlim=None, extend=0.03):
        """
        Plots 2D trajectories with one color per body and saves image in file
        """

        cdef Py_ssize_t ib, iint 
        cdef Py_ssize_t il, loop_id

        assert self.geodim == 2
        assert segmpos.shape[1] == self.segm_store
        
        if color_list is None:
            color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ncol = len(color_list)

        if xlim is None:

            xmin_arr, xmax_arr = self.DetectXlim(segmpos)

            xmin = xmin_arr[0]
            xmax = xmax_arr[0]
            ymin = xmin_arr[1]
            ymax = xmax_arr[1]

        else :

            xmin = xlim[0]
            xmax = xlim[1]
            ymin = xlim[2]
            ymax = xlim[3]
        
        xinf = xmin - extend*(xmax-xmin)
        xsup = xmax + extend*(xmax-xmin)
        
        yinf = ymin - extend*(ymax-ymin)
        ysup = ymax + extend*(ymax-ymin)
        
        hside = max(xsup-xinf,ysup-yinf)/2

        xmid = (xinf+xsup)/2
        ymid = (yinf+ysup)/2

        xinf = xmid - hside
        xsup = xmid + hside

        yinf = ymid - hside
        ysup = ymid + hside

        # Plot-related
        fig = plt.figure()
        fig.set_size_inches(fig_size)
        fig.set_dpi(dpi)
        ax = plt.gca()

        pos = np.empty((self.segm_size+1, self.geodim), dtype=np.float64)

        iplt = 0
        for ib in range(self.nbody):
            for iint in range(self.nint_min):
                if self._SegmRequiresDisp[ib,iint]:

                    iplt += 1

                    Sym = self.intersegm_to_all[ib][iint]
                    isegm = self._bodysegm[ib, iint]
                    Sym.TransformSegment(segmpos[isegm,:self.segm_size,:], pos[:self.segm_size,:])   

                    iintp = (iint+1)%self.nint_min
                    Sym = self.intersegm_to_all[ib][iintp]
                    isegmp = self._bodysegm[ib, iintp]
                    if Sym.TimeRev > 0:
                        Sym.TransformPos(segmpos[isegmp, 0,:], pos[self.segm_size,:])
                    else:
                        Sym.TransformPos(segmpos[isegmp,-2,:], pos[self.segm_size,:])

                    if (color is None) or (color == "none"):
                        current_color = color_list[0]
                    elif (color == "body"):
                        current_color = color_list[ib%ncol]
                    elif (color == "loop"):
                        current_color = color_list[self._bodyloop[ib]%ncol]
                    elif (color == "loop_id"):
                        loop_id = 0
                        il = self._bodyloop[ib]
                        while self._Targets[il,loop_id] != ib:
                            loop_id += 1
                        current_color = color_list[loop_id%ncol]
                    else:
                        raise ValueError(f'Unknown color scheme "{color}".')

                    plt.plot(pos[:,0], pos[:,1], color=current_color, antialiased=True, zorder=-iplt)

        ax.axis('off')
        ax.set_xlim([xinf, xsup])
        ax.set_ylim([yinf, ysup ])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        plt.savefig(filename)
        
        plt.close()

    @cython.final
    def PlotTimeBodyGraph(self, filename):
        PlotTimeBodyGraph(self.SegmGraph, self.nbody, self.nint_min, filename)

    @cython.final
    @cython.cdivision(True)
    def Make_params_bounds(self, double coeff_ampl_o=1e-1, Py_ssize_t k_infl=1, Py_ssize_t k_max=200, double coeff_ampl_min=1e-16):

        cdef double[::1] pos_buf_min = np.zeros((self.nparams_incl_o), dtype=np.float64)
        cdef double[::1] pos_buf_max = np.zeros((self.nparams_incl_o), dtype=np.float64)

        Make_Init_bounds_coeffs(
            &pos_buf_min[0]         , self._params_shapes   , self._params_shifts   ,
            self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
            self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
            self._ncoeff_min_loop   ,
            -coeff_ampl_o           , coeff_ampl_min        ,
            k_infl                  , k_max                 ,
        )

        Make_Init_bounds_coeffs(
            &pos_buf_max[0]         , self._params_shapes   , self._params_shifts   ,
            self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
            self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
            self._ncoeff_min_loop   ,
            coeff_ampl_o            , coeff_ampl_min        ,
            k_infl                  , k_max                 ,
        )

        mom_buf_min = self.params_changevar(pos_buf_min, inv=True, transpose=False)
        mom_buf_max = self.params_changevar(pos_buf_max, inv=True, transpose=False)

        return mom_buf_min, mom_buf_max

    @cython.final
    def params_changevar(self, double[::1] params_buf_in, bint inv=False, bint transpose=False):

        cdef double[::1] params_buf_out
        cdef double** params_pos_buf = <double**> malloc(sizeof(double*)*self.nloop)
        cdef Py_ssize_t il
        
        if inv:

            if transpose:

                assert params_buf_in.shape[0] == self.nparams
                params_buf_out = np.empty((self.nparams_incl_o), dtype=np.float64)

                for il in range(self.nloop):
                    params_pos_buf[il] = &params_buf_out[2*self._params_shifts[il]]
            
                changevar_mom_pos_invT(
                    &params_buf_in[0]       , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    params_pos_buf          , 
                )   

            else:

                assert params_buf_in.shape[0] == self.nparams_incl_o
                params_buf_out = np.empty((self.nparams), dtype=np.float64)

                for il in range(self.nloop):
                    params_pos_buf[il] = &params_buf_in[2*self._params_shifts[il]]

                changevar_mom_pos_inv(
                    params_pos_buf          , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    &params_buf_out[0]      , 
                )   

        else:

            if transpose:

                assert params_buf_in.shape[0] == self.nparams_incl_o
                params_buf_out = np.empty((self.nparams), dtype=np.float64)

                for il in range(self.nloop):
                    params_pos_buf[il] = &params_buf_in[2*self._params_shifts[il]]

                changevar_mom_pos_T(
                    params_pos_buf          , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    &params_buf_out[0]      , 
                )   

            else:

                assert params_buf_in.shape[0] == self.nparams
                params_buf_out = np.empty((self.nparams_incl_o), dtype=np.float64)

                for il in range(self.nloop):
                    params_pos_buf[il] = &params_buf_out[2*self._params_shifts[il]]

                changevar_mom_pos(
                    &params_buf_in[0]       , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    params_pos_buf          , 
                )   

        free(params_pos_buf)

        return np.asarray(params_buf_out)

    @cython.final
    def params_resize(self, double[::1] params_buf_in, Py_ssize_t nint_fac=1):

        assert params_buf_in.shape[0] == self.nparams

        cdef Py_ssize_t out_nint = 2 * self.nint_min * nint_fac
        cdef Py_ssize_t il

        params_shapes_list = []
        for il in range(self.nloop):

            nppl = self._params_basis_shapes[il,2]
            npr = out_nint // (2*self._ncoeff_min_loop[il])

            params_shapes_list.append((npr, self._nnz_k_shapes[il,0], nppl))

        cdef Py_ssize_t[:,::1] params_shapes_out
        cdef Py_ssize_t[::1] params_shifts_out
        params_shapes_out, params_shifts_out = BundleListOfShapes(params_shapes_list)

        cdef Py_ssize_t nparams_out = params_shifts_out[self.nloop] - self.nrem

        params_buf_out_np = np.zeros((nparams_out), dtype=np.float64)
        cdef double[::1] params_buf_out = params_buf_out_np

        cdef double* source = &params_buf_in[0]
        cdef double* dest = &params_buf_out[0]
        cdef int n_in, n_out

        if nint_fac < self._nint_fac:

            for il in range(self.nloop):

                n_out = params_shifts_out[il+1] - (params_shifts_out[il] + self._nco_in_loop[il])
                n_in = self._params_shifts[il+1] - (self._params_shifts[il] + self._nco_in_loop[il])

                scipy.linalg.cython_blas.dcopy(&n_out, source, &int_one, dest, &int_one)

                dest += n_out
                source += n_in

        else:

            for il in range(self.nloop):

                n_out = params_shifts_out[il+1] - (params_shifts_out[il] + self._nco_in_loop[il])
                n_in = self._params_shifts[il+1] - (self._params_shifts[il] + self._nco_in_loop[il])

                scipy.linalg.cython_blas.dcopy(&n_in, source, &int_one, dest, &int_one)

                dest += n_out
                source += n_in

        return params_buf_out_np

    @cython.final
    def all_coeffs_to_kin_nrg(self, double complex[:,:,::1] all_coeffs):

        cdef double kin = 0.
        cdef double fac, a
        cdef Py_ssize_t il, k, k2
        cdef int n = 2*self.geodim
        cdef double *loc

        for il in range(self.nloop):
            
            fac = ctwopisq * self._loopmass[il] * self._loopnb[il]

            for k in range(1,self.ncoeffs-1):

                k2 = k*k
                a = fac*k2
                
                loc = <double*> &all_coeffs[il,k,0]

                kin += a * scipy.linalg.cython_blas.ddot(&n, loc, &int_one, loc, &int_one)

        return kin

    @cython.final
    def all_coeffs_to_kin_nrg_grad(self, double complex[:,:,::1] all_coeffs):

        kin_grad_np = np.zeros((self.nloop, self.ncoeffs, self.geodim), dtype=np.complex128)
        cdef double complex [:,:,::1] kin_grad = kin_grad_np
        cdef double fac, a
        cdef Py_ssize_t il, k, k2
        cdef int n = 2*self.geodim
        cdef double *loc
        cdef double *grad_loc

        for il in range(self.nloop):
            
            fac = cfourpisq * self._loopmass[il] * self._loopnb[il]

            for k in range(1,self.ncoeffs-1):

                k2 = k*k
                a = fac*k2
                
                loc = <double*> &all_coeffs[il,k,0]
                grad_loc = <double*> &kin_grad[il,k,0]

                scipy.linalg.cython_blas.daxpy(&n, &a, loc, &int_one, grad_loc, &int_one)

        return kin_grad_np

    @cython.final
    def params_to_kin_nrg(self, double[::1] params_mom_buf):

        return params_to_kin_nrg(
            &params_mom_buf[0]  , self._params_shapes   , self._params_shifts   ,
            self._ncor_loop     , self._nco_in_loop     ,
        )

    @cython.final
    def params_to_kin_nrg_grad(self, double[::1] params_mom_buf):

        grad_buf_np = np.zeros((self.nparams), dtype=np.float64)
        cdef double[::1] grad_buf = grad_buf_np

        params_to_kin_nrg_grad_daxpy(
            &params_mom_buf[0]  , self._params_shapes   , self._params_shifts   ,
            self._ncor_loop     , self._nco_in_loop     ,
            1.                  ,
            &grad_buf[0]        ,
        )

        return grad_buf_np
        
    @cython.final
    def segmpos_to_hash(self, double[:,:,::1] segmpos):

        assert segmpos.shape[1] == self.segm_store

        Hash_np = np.empty((self._Hash_exp.shape[0]), dtype=np.float64)
        cdef double[::1] Hash = Hash_np

        with nogil:

            segm_pos_to_hash(
                segmpos                 ,
                self._BinSourceSegm     , self._BinTargetSegm   ,
                self._BinSpaceRot       , self._BinSpaceRotIsId ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store       ,
                self._Hash_exp          , Hash                  ,   
            )
        
        return Hash_np

    @cython.final
    def params_to_pot_nrg(self, double[::1] params_mom_buf):

        assert params_mom_buf.shape[0] == self.nparams

        cdef double pot_nrg
        cdef double[:,:,::1] segmpos = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)

        with nogil:

            params_to_segmpos(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                segmpos                     ,
            )

            pot_nrg = segm_pos_to_pot_nrg(
                segmpos                 ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           ,
                self._inter_law         , self._inter_law_param_ptr ,
            )
        
        return pot_nrg

    @cython.final
    def params_to_pot_nrg_grad(self, double[::1] params_mom_buf):

        assert params_mom_buf.shape[0] == self.nparams

        params_grad_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] params_grad = params_grad_np

        with nogil:

            params_to_segmpos(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                self._segmpos               ,
            )

            memset(&self._pot_nrg_grad[0,0,0], 0, sizeof(double)*self.nsegm*self.segm_store*self.geodim)

            segm_pos_to_pot_nrg_grad(
                self._segmpos           , self._pot_nrg_grad        ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           , 1.    ,
                self._inter_law         , self._inter_law_param_ptr ,
            )

            segmpos_to_params_T(
                self._pot_nrg_grad          ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                params_grad                 ,
            )

        return params_grad_np

    @cython.final
    def params_to_pot_nrg_hess(self, double[::1] params_mom_buf, double[::1] dparams_mom_buf):

        # Not actually used in optimization to allow for segmpos caching.
        # This explains the non pre-allocation of dsegmpos

        assert params_mom_buf.shape[0] == self.nparams
        assert dparams_mom_buf.shape[0] == self.nparams

        params_hess_np = np.empty((self.nparams), dtype=np.float64)

        cdef double[:,:,::1] dsegmpos = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)
        cdef double[::1] params_hess = params_hess_np

        with nogil:

            params_to_segmpos(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                self._segmpos               ,
            )

            params_to_segmpos(
                dparams_mom_buf             ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                dsegmpos                    ,
            )

            memset(&self._pot_nrg_grad[0,0,0], 0, sizeof(double)*self.nsegm*self.segm_store*self.geodim)

            segm_pos_to_pot_nrg_hess(
                self._segmpos           , dsegmpos                  , self._pot_nrg_grad        ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           , 1.                        ,
                self._inter_law         , self._inter_law_param_ptr ,
            )

            segmpos_to_params_T(
                self._pot_nrg_grad          ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                params_hess                 ,
            )

        return params_hess_np
    
    @cython.final
    def params_to_action(self, double[::1] params_mom_buf):

        assert params_mom_buf.shape[0] == self.nparams

        cdef double action

        with nogil:

            action = params_to_kin_nrg(
                &params_mom_buf[0]      , self._params_shapes       , self._params_shifts       ,
                self._ncor_loop         , self._nco_in_loop         ,
            )

            params_to_segmpos(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                self._segmpos               ,
            )

            action -= segm_pos_to_pot_nrg(
                self._segmpos           ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           ,
                self._inter_law         , self._inter_law_param_ptr ,
            )
        
        return action

    @cython.final
    def params_to_action_grad(self, double[::1] params_mom_buf):
        """ _summary_

        _extended_summary_

        Parameters
        ----------
        double : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        assert params_mom_buf.shape[0] == self.nparams

        action_grad_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] action_grad = action_grad_np

        with nogil:

            params_to_segmpos(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                self._segmpos               ,
            )

            memset(&self._pot_nrg_grad[0,0,0], 0, sizeof(double)*self.nsegm*self.segm_store*self.geodim)

            segm_pos_to_pot_nrg_grad(
                self._segmpos           , self._pot_nrg_grad        ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           , -1.   ,
                self._inter_law         , self._inter_law_param_ptr ,
            )

            segmpos_to_params_T(
                self._pot_nrg_grad          ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                action_grad                 ,
            )

            params_to_kin_nrg_grad_daxpy(
                &params_mom_buf[0]  , self._params_shapes   , self._params_shifts   ,
                self._ncor_loop     , self._nco_in_loop     ,
                1.                  ,
                &action_grad[0]     ,
            )

        return action_grad_np

    @cython.final
    def params_to_action_hess(self, double[::1] params_mom_buf, double[::1] dparams_mom_buf):

        # Not actually used in optimization loop to allow for segmpos caching.
        # This explains the non pre-allocation of dsegmpos

        assert params_mom_buf.shape[0] == self.nparams
        assert dparams_mom_buf.shape[0] == self.nparams

        cdef double[:,:,::1] dsegmpos = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)
        action_hess_np =  np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] action_hess = action_hess_np

        with nogil:

            params_to_segmpos(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                self._segmpos               ,
            )

            params_to_segmpos(
                dparams_mom_buf             ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                dsegmpos                    ,
            )

            memset(&self._pot_nrg_grad[0,0,0], 0, sizeof(double)*self.nsegm*self.segm_store*self.geodim)

            segm_pos_to_pot_nrg_hess(
                self._segmpos           , dsegmpos                  , self._pot_nrg_grad        ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           , -1.                       ,
                self._inter_law         , self._inter_law_param_ptr ,
            )

            segmpos_to_params_T(
                self._pot_nrg_grad          ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                action_hess                 ,
            )

            params_to_kin_nrg_grad_daxpy(
                &dparams_mom_buf[0] , self._params_shapes   , self._params_shifts   ,
                self._ncor_loop     , self._nco_in_loop     ,
                1.                  ,
                &action_hess[0]     ,
            )

        return action_hess_np
    
    @cython.final
    def segmpos_params_to_action(self, double[:,:,::1] segmpos, double[::1] params_mom_buf):
        
        assert segmpos.shape[1] == self.segm_store
        assert params_mom_buf.shape[0] == self.nparams

        cdef double action

        with nogil:

            action = params_to_kin_nrg(
                &params_mom_buf[0]      , self._params_shapes       , self._params_shifts       ,
                self._ncor_loop         , self._nco_in_loop         ,
            )

            action -= segm_pos_to_pot_nrg(
                segmpos                 ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           ,
                self._inter_law         , self._inter_law_param_ptr ,
            )
        
        return action

    @cython.final
    def segmpos_params_to_action_grad(self, double[:,:,::1] segmpos, double[::1] params_mom_buf):
        
        assert segmpos.shape[1] == self.segm_store
        assert params_mom_buf.shape[0] == self.nparams

        action_grad_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] action_grad = action_grad_np

        with nogil:

            memset(&self._pot_nrg_grad[0,0,0], 0, sizeof(double)*self.nsegm*self.segm_store*self.geodim)

            segm_pos_to_pot_nrg_grad(
                segmpos                 , self._pot_nrg_grad        ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           , -1.   ,
                self._inter_law         , self._inter_law_param_ptr ,
            )

            segmpos_to_params_T(
                self._pot_nrg_grad          ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                action_grad                 ,
            )

            params_to_kin_nrg_grad_daxpy(
                &params_mom_buf[0]  , self._params_shapes   , self._params_shifts   ,
                self._ncor_loop     , self._nco_in_loop     ,
                1.                  ,
                &action_grad[0]     ,
            )

        return action_grad_np

    @cython.final
    def segmpos_dparams_to_action_hess(self, double[:,:,::1] segmpos, double[::1] dparams_mom_buf):

        assert segmpos.shape[1] == self.segm_store
        assert dparams_mom_buf.shape[0] == self.nparams

        action_hess_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] action_hess = action_hess_np

        with nogil:

            params_to_segmpos(
                dparams_mom_buf             ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                self._segmpos               , # self._segmpos is actually dsegmpos. Prevents allocation of a new buffer.
            )

            memset(&self._pot_nrg_grad[0,0,0], 0, sizeof(double)*self.nsegm*self.segm_store*self.geodim)

            segm_pos_to_pot_nrg_hess(
                segmpos                 , self._segmpos             , self._pot_nrg_grad        ,
                self._BinSourceSegm     , self._BinTargetSegm       ,
                self._BinSpaceRot       , self._BinSpaceRotIsId     ,
                self._BinProdChargeSum  ,
                self.segm_size          , self.segm_store           , -1.                       ,
                self._inter_law         , self._inter_law_param_ptr ,
            )

            segmpos_to_params_T(
                self._pot_nrg_grad          ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                action_hess                 ,
            )

            params_to_kin_nrg_grad_daxpy(
                &dparams_mom_buf[0] , self._params_shapes   , self._params_shifts   ,
                self._ncor_loop     , self._nco_in_loop     ,
                1.                  ,
                &action_hess[0]     ,
            )

        return action_hess_np

    @cython.final
    def params_to_all_coeffs_noopt(self, double[::1] params_mom_buf, bint transpose=False, double dt = 0.):

        all_coeffs_dense = self.params_to_all_coeffs_dense_noopt(params_mom_buf, transpose, dt)
        
        all_coeffs = np.zeros((self.nloop, self.ncoeffs, self.geodim), dtype=np.complex128)

        for il in range(self.nloop):
            
            nnz_k = self.nnz_k(il)

            assert (self.ncoeffs-1) % self._ncoeff_min_loop[il] == 0

            npr = (self.ncoeffs-1) //  self._ncoeff_min_loop[il]

            coeffs_dense = all_coeffs[il,:(self.ncoeffs-1),:].reshape(npr, self._ncoeff_min_loop[il], self.geodim)                
            coeffs_dense[:,nnz_k,:] = all_coeffs_dense[il]

        all_coeffs[:,0,:].imag = 0

        return all_coeffs    

    @cython.cdivision(True)
    @cython.final
    def params_to_all_coeffs_dense_noopt(self, double[::1] params_mom_buf, bint transpose=False, double dt = 0.):

        assert params_mom_buf.shape[0] == self.nparams

        cdef Py_ssize_t il
        cdef Py_ssize_t ikp, ikr, k, kmod
        cdef double alpha, arg, alpha_dt
        cdef double complex w
        cdef int geodim = self.geodim

        cdef Py_ssize_t[::1] nnz_k 
        cdef double complex[:,:,::1] coeffs_dense
        cdef np.ndarray[double, ndim=1, mode='c'] params_pos_buf_np = np.empty((self.nparams_incl_o), dtype=np.float64)
        cdef double** params_pos_buf = <double**> malloc(sizeof(double*)*self.nloop)

        for il in range(self.nloop):
            params_pos_buf[il] = &params_pos_buf_np[2*self._params_shifts[il]]
    
        if transpose:
            changevar_mom_pos_invT(
                &params_mom_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                params_pos_buf          , 
            )   
        else:
            changevar_mom_pos(
                &params_mom_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                params_pos_buf          , 
            )   

        free(params_pos_buf)

        all_coeffs_dense = []
        for il in range(self.nloop):
            
            params_basis = self.params_basis_pos(il)
            nnz_k = self.nnz_k(il)

            shape = np.asarray(self._params_shapes[il]).copy()
            shape[0] *= 2
            
            params_loop = params_pos_buf_np[2*self._params_shifts[il]:2*self._params_shifts[il+1]].reshape(shape)

            coeffs_dense = np.einsum('ijk,ljk->lji', params_basis, params_loop[:self._params_shapes[il,0],:,:]).copy()

            arg = self._gensegm_loop_start[il]
            alpha =  - ctwopi * (arg / self.nint_min)
            alpha_dt =  - ctwopi * dt # Beware of loss of precision
            
            if nnz_k.shape[0] > 0:

                for ikp in range(coeffs_dense.shape[0]):
                    for ikr in range(coeffs_dense.shape[1]):

                        k = (ikp * self._ncoeff_min_loop[il] + nnz_k[ikr])
                        kmod = k % self.nint_min

                        arg = alpha * kmod + alpha_dt*k
                        w = ccos(arg) + 1j*csin(arg)
                        scipy.linalg.cython_blas.zscal(&geodim,&w,&coeffs_dense[ikp,ikr,0],&int_one)

            all_coeffs_dense.append(coeffs_dense)

        return all_coeffs_dense      

    @cython.final
    def all_coeffs_to_params_noopt(self, all_coeffs, bint transpose = False, double dt = 0.):

        assert all_coeffs.shape[0] == self.nloop
        assert all_coeffs.shape[1] == self.ncoeffs
        assert all_coeffs.shape[2] == self.geodim

        all_coeffs_dense = []

        for il in range(self.nloop):

            nnz_k = self.nnz_k(il)

            npr = (self.ncoeffs-1) //  self._ncoeff_min_loop[il]

            coeffs_dense = all_coeffs[il,:(self.ncoeffs-1),:].reshape(npr, self._ncoeff_min_loop[il], self.geodim)[:,nnz_k,:].copy()
            all_coeffs_dense.append(coeffs_dense)

        return self.all_coeffs_dense_to_params_noopt(all_coeffs_dense, transpose, dt)

    @cython.cdivision(True)
    @cython.final
    def all_coeffs_dense_to_params_noopt(self, all_coeffs_dense, bint transpose = False, double dt = 0.):

        cdef Py_ssize_t il
        cdef Py_ssize_t ikp, ikr, k, kmod
        cdef double alpha, arg, alpha_dt
        cdef double complex w
        cdef int geodim = self.geodim

        cdef Py_ssize_t[::1] nnz_k 
        cdef double complex[:,:,::1] coeffs_dense
        cdef np.ndarray[double, ndim=1, mode='c'] params_pos_buf_np = np.empty((self.nparams_incl_o), dtype=np.float64)
        
        cdef double** params_pos_buf = <double**> malloc(sizeof(double*)*self.nloop)
        for il in range(self.nloop):
            params_pos_buf[il] = &params_pos_buf_np[2*self._params_shifts[il]]

        for il in range(self.nloop):

            params_basis = self.params_basis_pos(il)
            nnz_k = self.nnz_k(il)

            coeffs_dense = all_coeffs_dense[il]
            
            arg = self._gensegm_loop_start[il]
            alpha = ctwopi * arg / self.nint_min
            alpha_dt = ctwopi * dt # Beware of loss of precision
            
            if nnz_k.shape[0] > 0:

                for ikp in range(coeffs_dense.shape[0]):
                    for ikr in range(coeffs_dense.shape[1]):

                        k = (ikp * self._ncoeff_min_loop[il] + nnz_k[ikr])
                        kmod = k % self.nint_min
                        
                        arg = alpha * kmod + alpha_dt*k
                        w = ccos(arg) + 1j*csin(arg)
                        scipy.linalg.cython_blas.zscal(&geodim,&w,&coeffs_dense[ikp,ikr,0],&int_one)

            shape = np.asarray(self._params_shapes[il]).copy()
            shape[0] *= 2             

            params_loop = params_pos_buf_np[2*self._params_shifts[il]:2*self._params_shifts[il+1]].reshape(shape)

            params_loop[:self._params_shapes[il,0],:,:] = np.einsum('ijk,lji->ljk', params_basis.conj(), coeffs_dense).real

        params_mom_buf_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] params_mom_buf = params_mom_buf_np

        if transpose:
            changevar_mom_pos_T(
                &params_pos_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                &params_mom_buf[0]      , 
            )   
        else:
            changevar_mom_pos_inv(
                &params_pos_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                &params_mom_buf[0]      , 
            )   

        free(params_pos_buf)

        return params_mom_buf_np   

    @cython.final
    def all_pos_to_all_body_pos_noopt(self, all_pos):

        assert all_pos.shape[0] == self._nint
        
        all_body_pos = np.zeros((self.nbody, self._nint, self.geodim), dtype=np.float64)

        for ib in range(self.nbody):
            
            il = self._bodyloop[ib]
            ib_gen = self._loopgen[il]
            
            if (ib == ib_gen) :
                
                all_body_pos[ib,:,:] = all_pos[il,:,:]
                
            else:
            
                path = networkx.shortest_path(self.BodyGraph, source = ib_gen, target = ib)

                pathlen = len(path)

                TotSym = ActionSym.Identity(self.nbody, self.geodim)
                for ipath in range(1,pathlen):

                    if (path[ipath-1] > path[ipath]):
                        Sym = self.BodyGraph.edges[(path[ipath], path[ipath-1])]["SymList"][0].Inverse()
                    else:
                        Sym = self.BodyGraph.edges[(path[ipath-1], path[ipath])]["SymList"][0]

                    TotSym = Sym.Compose(TotSym)
            
                for iint_gen in range(self._nint):
                    
                    tnum, tden = TotSym.ApplyT(iint_gen, self._nint)
                    iint_target = tnum * self._nint // tden
                    
                    all_body_pos[ib,iint_target,:] = np.matmul(TotSym.SpaceRot, all_pos[il,iint_gen,:])

        return all_body_pos     

    @cython.final
    def all_to_segm_noopt(self, all_pos, pos=True):
        # Ugly code.
        
        assert self._nint == all_pos.shape[1]
        
        cdef Py_ssize_t ib, iint, il

        segmvals = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)

        for isegm in range(self.nsegm):

            ib = self._gensegm_to_body[isegm]
            iint = self._gensegm_to_iint[isegm]
            il = self._bodyloop[ib]

            assert isegm == self._bodysegm[ib,iint]

            ibeg = iint * self.segm_size         
            iend = ibeg + self.segm_store

            if iend <= self._nint:

                if self._InterTimeRev[isegm] > 0:

                    np.matmul(
                        all_pos[il,ibeg:iend,:]         ,
                        self._InterSpaceRotPos[isegm,:,:]  ,
                        out = segmvals[isegm,:,:]       ,
                    )            

                else:

                    if pos:
                        segmvals[isegm,:,:] = np.matmul(
                            all_pos[il,ibeg:iend,:]         ,
                            self._InterSpaceRotPos[isegm,:,:]  ,
                        )[::-1,:]
                    else:
                        segmvals[isegm,:,:] = -np.matmul(
                            all_pos[il,ibeg:iend,:]         ,
                            self._InterSpaceRotPos[isegm,:,:]  ,
                        )[::-1,:]
            else:

                iend = iend - 1

                assert iend == self._nint

                if self._InterTimeRev[isegm] > 0:

                    if pos:
                        np.matmul(
                            all_pos[il,ibeg:iend,:]         ,
                            self._InterSpaceRotPos[isegm,:,:]  ,
                            out = segmvals[isegm,:self.segm_store-1,:]        ,
                        )   

                        np.matmul(
                            all_pos[il,0,:]         ,
                            self._InterSpaceRotPos[isegm,:,:]  ,
                            out = segmvals[isegm,self.segm_store-1,:]        ,
                        )   

                    else:     
                        np.matmul(
                            all_pos[il,ibeg:iend,:]         ,
                            self._InterSpaceRotVel[isegm,:,:]  ,
                            out = segmvals[isegm,:self.segm_store-1,:]        ,
                        )   

                        np.matmul(
                            all_pos[il,0,:]         ,
                            self._InterSpaceRotVel[isegm,:,:]  ,
                            out = segmvals[isegm,self.segm_store-1,:]        ,
                        )       

                else:

                    if pos:
                        segmvals[isegm,1:,:] = np.matmul(
                            all_pos[il,ibeg:iend,:]         ,
                            self._InterSpaceRotPos[isegm,:,:]  ,
                        )[::-1,:]

                        segmvals[isegm,0,:] = np.matmul(
                            all_pos[il,0,:]         ,
                            self._InterSpaceRotPos[isegm,:,:]  ,
                        )

                    else:
                        segmvals[isegm,1:,:] = np.matmul(
                            all_pos[il,ibeg:iend,:]         ,
                            self._InterSpaceRotVel[isegm,:,:]  ,
                        )[::-1,:]

                        segmvals[isegm,0,:] = np.matmul(
                            all_pos[il,0,:]         ,
                            self._InterSpaceRotVel[isegm,:,:]  ,
                        )

        return segmvals

    @cython.final
    def segmpos_to_all_noopt(self, double[:,:,::1] segmpos, bint pos=True):

        assert self.segm_store == segmpos.shape[1]

        cdef Py_ssize_t ib, iint, il
        cdef Py_ssize_t ibeg, segmend
        cdef Py_ssize_t segmbeg, iend
        cdef ActionSym Sym

        cdef np.ndarray[double, ndim=3, mode='c'] all_pos = np.empty((self.nloop, self._nint, self.geodim), dtype=np.float64)

        for il in range(self.nloop):

            ib = self._loopgen[il]

            for iint in range(self.nint_min):

                isegm = self._bodysegm[ib, iint]

                if pos:
                    Sym = self.intersegm_to_all[ib][iint]
                else:
                    Sym = self.intersegm_to_all[ib][iint].TimeDerivative()

                ibeg = iint * self.segm_size         
                iend = ibeg + self.segm_size
                assert iend <= self._nint

                if Sym.TimeRev > 0:
                    segmbeg = 0
                    segmend = self.segm_size
                else:
                    segmbeg = 1
                    segmend = self.segm_size+1

                Sym.TransformSegment(segmpos[isegm,segmbeg:segmend,:], all_pos[il,ibeg:iend,:])

        return all_pos
        
    @cython.final
    def segmpos_to_allbody_noopt(self, double[:,:,::1] segmpos, bint pos=True):

        assert self.segm_store == segmpos.shape[1]

        cdef Py_ssize_t ib, iint
        cdef Py_ssize_t ibeg, segmend
        cdef Py_ssize_t segmbeg, iend
        cdef ActionSym Sym

        cdef double[:,:,::1] all_bodypos = np.empty((self.nbody, self._nint, self.geodim), dtype=np.float64)

        for ib in range(self.nbody):

            for iint in range(self.nint_min):

                isegm = self._bodysegm[ib, iint]

                if pos:
                    Sym = self.intersegm_to_all[ib][iint]
                else:
                    Sym = self.intersegm_to_all[ib][iint].TimeDerivative()

                ibeg = iint * self.segm_size         
                iend = ibeg + self.segm_size
                assert iend <= self._nint

                if Sym.TimeRev > 0:
                    segmbeg = 0
                    segmend = self.segm_size
                else:
                    segmbeg = 1
                    segmend = self.segm_size+1

                Sym.TransformSegment(segmpos[isegm,segmbeg:segmend,:], all_bodypos[ib,ibeg:iend,:])

        return all_bodypos

    @cython.cdivision(True)
    @cython.final
    def ComputeSymDefault(self, double[:,:,::1] segmpos, ActionSym Sym, Py_ssize_t lnorm = 1, full = True, pos = True):

        if lnorm not in [1,2,22]:
            raise ValueError(f'ComputeSymDefault only computes L1, L2 or L2 squared norms. Received {lnorm = }.')

        cdef Py_ssize_t ib , iint
        cdef Py_ssize_t ibp, iintp
        cdef Py_ssize_t segmbeg, segmend
        cdef Py_ssize_t isegm
        cdef ActionSym CSym  
        cdef int size = self.geodim * self.segm_size
        cdef np.ndarray[double, ndim=2, mode='c'] trans_pos  = np.empty((self.segm_size, self.geodim), dtype=np.float64)
        cdef np.ndarray[double, ndim=2, mode='c'] trans_posp = np.empty((self.segm_size, self.geodim), dtype=np.float64)

        cdef double res = 0

        if full:
            all_iints = range(self.nint_min)
            n_pts = self.nbody * self.nint_min * self.segm_size
        else:
            all_iints = [0]
            n_pts = self.nbody * self.segm_size

        for ib in range(self.nbody):

            for iint in all_iints:

                # Computing trans_pos
                isegm = self._bodysegm[ib, iint]
                if pos:
                    CSym = Sym.Compose(self.intersegm_to_all[ib][iint])
                else:
                    CSym = Sym.Compose(self.intersegm_to_all[ib][iint].TimeDerivative())

                if CSym.TimeRev > 0:
                    segmbeg = 0
                    segmend = self.segm_size
                else:
                    segmbeg = 1
                    segmend = self.segm_size+1
                    assert self.GreaterNStore

                CSym.TransformSegment(segmpos[isegm,segmbeg:segmend,:], trans_pos)

                # Computing trans_posp

                ibp = Sym._BodyPerm[ib]
                tnum_target, tden_target = Sym.ApplyTSegm(iint, self.nint_min)
                assert self.nint_min % tden_target == 0
                iintp = (tnum_target * (self.nint_min // tden_target) + self.nint_min) % self.nint_min

                isegmp = self._bodysegm[ibp, iintp]
                if pos:
                    CSym = self.intersegm_to_all[ibp][iintp]
                else:
                    CSym = self.intersegm_to_all[ibp][iintp].TimeDerivative()

                if CSym.TimeRev > 0:
                    segmbeg = 0
                    segmend = self.segm_size
                else:
                    segmbeg = 1
                    segmend = self.segm_size+1
                    assert self.GreaterNStore

                CSym.TransformSegment(segmpos[isegmp,segmbeg:segmend,:], trans_posp)

                scipy.linalg.cython_blas.daxpy(&size,&minusone_double,&trans_posp[0,0],&int_one,&trans_pos[0,0],&int_one)
                if lnorm == 1:
                    res += scipy.linalg.cython_blas.dasum(&size,&trans_pos[0,0],&int_one)
                else:
                    res += scipy.linalg.cython_blas.ddot(&size, &trans_pos[0,0], &int_one, &trans_pos[0,0], &int_one)

        if lnorm == 1:
            return res / n_pts
        elif lnorm == 2:
            return csqrt(res / n_pts)
        else:
            return res / n_pts

    @cython.final
    def params_to_segmpos(self, double[::1] params_mom_buf):

        assert params_mom_buf.shape[0] == self.nparams

        segmpos_np = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)
        cdef double[:,:,::1] segmpos = segmpos_np

        with nogil:

            params_to_segmpos(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                segmpos                     ,
            )

        return segmpos_np        

    @cython.final
    def params_to_segmvel(self, double[::1] params_mom_buf):

        assert params_mom_buf.shape[0] == self.nparams

        segmvel_np = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)
        cdef double[:,:,::1] segmvel = segmvel_np

        with nogil:

            params_to_segmvel(
                params_mom_buf              ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_vel  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutVel ,
                self._fftw_genrfft_exe      , self._fftw_symirfft_exe   ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotVelIsId  , self._InterSpaceRotVel    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotVel     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                segmvel                     ,
            )

        return segmvel_np

    def inplace_segmvel_to_segmmom(self, double[:,:,:] segmvel):

        cdef double mass
        cdef int n = self.segm_store * self.geodim
        cdef Py_ssize_t isegm

        for isegm in range(self.nsegm):
            mass = self._loopmass[self._bodyloop[self._intersegm_to_body[isegm]]]
            scipy.linalg.cython_blas.dscal(&n,&mass,&segmvel[isegm,0,0],&int_one)

    @cython.final
    def params_to_segmmom(self, double[::1] params_mom_buf):

        segmom = self.params_to_segmvel(params_mom_buf)
        self.inplace_segmvel_to_segmmom(segmom)

        return segmom

    @cython.final
    def segmpos_to_params(self, double[:,:,::1] segmpos):

        assert self.segm_store == segmpos.shape[1]

        params_mom_buf_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] params_mom_buf = params_mom_buf_np

        with nogil:

            segmpos_to_params(
                segmpos                     ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                params_mom_buf              ,
            )

        return params_mom_buf_np    

    @cython.final
    def segmpos_to_params_T(self, double[:,:,::1] segmpos):

        assert self.segm_store == segmpos.shape[1]

        params_mom_buf_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] params_mom_buf = params_mom_buf_np

        with nogil:

            segmpos_to_params_T(
                segmpos                     ,
                self._params_pos_buf        , self._params_shapes       , self._params_shifts       ,
                self._ifft_buf_ptr          , self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf_pos  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf             , self._co_in_shapes        , self._co_in_shifts        ,
                self._pos_slice_buf_ptr     , self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop       , self._n_sub_fft           , self._fft_backend         ,
                self._ParamBasisShortcutPos ,
                self._fftw_genirfft_exe     , self._fftw_symrfft_exe    ,
                self._loopnb                , self._loopmass            ,
                self._InterSpaceRotPosIsId  , self._InterSpaceRotPos    , self._InterTimeRev        ,
                self._ALG_Iint              , self._ALG_SpaceRotPos     , self._ALG_TimeRev         ,
                self._gensegm_to_body       , self._gensegm_to_iintrel  ,
                self._bodyloop              , self.segm_size            , self.segm_store           ,
                params_mom_buf              ,
            )

        return params_mom_buf_np

    @cython.final
    def TT_params_to_action_grad(self, double[::1] params_mom_buf, object TT):
        """ Profiles the computation of the gradient of the action with respect to parameters.

        Parameters
        ----------
        params_mom_buf : :class:`numpy:numpy.ndarray`:class:`(shape = nparams, dtype = np.float64)`
            Buffer of parameters.
        TT : :class:`pyquickbench:pyquickbench.TimeTrain`
            Profiler object that records the following intermediate times:

            * start
            * memory
            * changevar_mom_pos
            * params_to_pos_slice
            * pos_slice_to_segmpos
            * segm_pos_to_pot_nrg_grad
            * segmpos_to_pos_slice_T
            * pos_slice_to_params
            * changevar_mom_pos_T
            * params_to_kin_nrg_grad_daxpy

        Returns
        -------
        :class:`numpy:numpy.ndarray`:class:`(shape = nparams, dtype = np.float64)`
            Gradient of the action with respect to parameters.

        See Also
        --------

        * :attr:`nparams`
        * :meth:`params_to_action_grad`
        * :mod:`pyquickbench:pyquickbench`

        """

        TT.toc("start")

        assert params_mom_buf.shape[0] == self.nparams

        action_grad_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] action_grad = action_grad_np

        TT.toc("memory")

        cdef int nsegm = self._gensegm_to_body.shape[0]
        cdef int geodim = self._InterSpaceRotPos.shape[1]

        changevar_mom_pos(
            &params_mom_buf[0]        , self._params_shapes , self._params_shifts ,
            self._nnz_k_buf           , self._nnz_k_shapes  , self._nnz_k_shifts  ,
            self._co_in_buf           , self._co_in_shapes  , self._co_in_shifts  ,
            self._ncoeff_min_loop     ,
            self._loopnb              , self._loopmass      ,
            self._params_pos_buf      , 
        )   

        TT.toc("changevar_mom_pos")

        params_to_pos_slice(
            self._params_pos_buf            , self._params_shapes       , self._params_shifts       ,
            &self._nnz_k_buf[0]             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
            self._ifft_buf_ptr              , self._ifft_shapes         , self._ifft_shifts         ,
            self._ParamBasisShortcutPos     ,
            self._fft_backend               , self._fftw_genrfft_exe    , self._fftw_symirfft_exe   ,
            &self._params_basis_buf_pos[0]  , self._params_basis_shapes , self._params_basis_shifts ,
            self._pos_slice_buf_ptr         , self._pos_slice_shapes    , self._pos_slice_shifts    ,
            self._ncoeff_min_loop           , self._n_sub_fft           ,
        )

        if (self.segm_size != self.segm_store):

            Adjust_after_last_gen(
                self._pos_slice_buf_ptr     , self._pos_slice_shifts      ,
                self._ifft_shapes           ,
                self._params_basis_shapes   ,
                self._n_sub_fft             ,
                self._ALG_Iint              ,
                self._ALG_TimeRev           , self._ALG_SpaceRotPos         ,
                self.segm_size              ,
            )

        TT.toc("params_to_pos_slice")

        pos_slice_to_segmpos(
            self._pos_slice_buf_ptr     , self._pos_slice_shapes  , self._pos_slice_shifts ,
            &self._segmpos[0,0,0]       ,
            self._InterSpaceRotPosIsId  ,
            self._InterSpaceRotPos      ,
            self._InterTimeRev          ,
            self._gensegm_to_body       ,
            self._gensegm_to_iintrel    ,
            self._bodyloop              ,
            self.segm_size              ,
            self.segm_store             ,
        )

        TT.toc("pos_slice_to_segmpos")

        memset(&self._pot_nrg_grad[0,0,0], 0, sizeof(double)*self.nsegm*self.segm_store*self.geodim)

        segm_pos_to_pot_nrg_grad(
            self._segmpos           , self._pot_nrg_grad        ,
            self._BinSourceSegm     , self._BinTargetSegm       ,
            self._BinSpaceRot       , self._BinSpaceRotIsId     ,
            self._BinProdChargeSum  ,
            self.segm_size          , self.segm_store           , -1.   ,
            self._inter_law         , self._inter_law_param_ptr ,
        )

        TT.toc("segm_pos_to_pot_nrg_grad")

        segmpos_to_pos_slice_T(
            &self._pot_nrg_grad[0,0,0]  ,
            self._pos_slice_buf_ptr     , self._pos_slice_shapes  , self._pos_slice_shifts ,
            self._InterSpaceRotPosIsId  ,
            self._InterSpaceRotPos      ,
            self._InterTimeRev          ,
            self._gensegm_to_body       ,
            self._gensegm_to_iintrel    ,
            self._bodyloop              ,
            self.segm_size              ,
            self.segm_store             ,
        )

        if (self.segm_size != self.segm_store):
            Adjust_after_last_gen_T(
                self._pos_slice_buf_ptr   , self._pos_slice_shifts      ,
                self._ifft_shapes         ,
                self._params_basis_shapes ,
                self._n_sub_fft           ,
                self._ALG_Iint            ,
                self._ALG_TimeRev         , self._ALG_SpaceRotPos       ,
                self.segm_size            , 
            )

        TT.toc("segmpos_to_pos_slice_T")

        pos_slice_to_params(
            self._pos_slice_buf_ptr         , self._pos_slice_shapes    , self._pos_slice_shifts    ,
            &self._params_basis_buf_pos[0]  , self._params_basis_shapes , self._params_basis_shifts ,
            &self._nnz_k_buf[0]             , self._nnz_k_shapes        , self._nnz_k_shifts        ,
            self._ifft_buf_ptr              , self._ifft_shapes         , self._ifft_shifts         ,
            self._ncoeff_min_loop           , self._n_sub_fft           , -1                        ,
            self._params_pos_buf            , self._params_shapes       , self._params_shifts       ,
            self._ParamBasisShortcutPos     ,
            self._fft_backend               , self._fftw_genirfft_exe   , self._fftw_symrfft_exe    ,
        )

        TT.toc("pos_slice_to_params")

        changevar_mom_pos_T(
            self._params_pos_buf      , self._params_shapes , self._params_shifts ,
            self._nnz_k_buf           , self._nnz_k_shapes  , self._nnz_k_shifts  ,
            self._co_in_buf           , self._co_in_shapes  , self._co_in_shifts  ,
            self._ncoeff_min_loop     ,
            self._loopnb              , self._loopmass      ,
            &params_mom_buf[0]  , 
        )   

        TT.toc("changevar_mom_pos_T")

        params_to_kin_nrg_grad_daxpy(
            &params_mom_buf[0]  , self._params_shapes   , self._params_shifts   ,
            self._ncor_loop     , self._nco_in_loop     ,
            1.                  ,
            &action_grad[0]     ,
        )

        TT.toc("params_to_kin_nrg_grad_daxpy")

        return action_grad_np

    @cython.final
    def segm_to_path_stats(self, double[:,:,::1] segmpos, double[:,:,::1] segmvel):

        cdef Py_ssize_t il, isegm, ib

        out_segm_len = np.empty((self.nsegm), dtype=np.float64)
        cdef double[::1] out_segm_len_mv = out_segm_len

        out_bin_dx_min = np.empty((self.nbin_segm_unique), dtype=np.float64)
        cdef double[::1] out_bin_dx_min_mv = out_bin_dx_min

        with nogil:
            
            segmpos_to_unary_path_stats(
                segmpos                 ,
                segmvel                 ,
                self.segm_size          ,
                self.segm_store         ,
                self.nint_min           ,
                out_segm_len_mv         ,
            )            

            segmpos_to_binary_path_stats(
                segmpos                 ,
                self._BinSourceSegm     , self._BinTargetSegm   ,
                self._BinSpaceRot       , self._BinSpaceRotIsId ,
                self.segm_store         ,
                out_bin_dx_min_mv       ,
            )

        out_loop_len = np.zeros((self.nloop), dtype=np.float64)
        cdef double[::1] out_loop_len_mv = out_loop_len

        for isegm in range(self.nsegm):
            ib = self._gensegm_to_body[isegm]
            il = self._bodyloop[ib]
            out_loop_len_mv[il] += out_segm_len_mv[isegm]

        for il in range(self.nloop):
            out_loop_len_mv[il] /= self.nint_min 

        return out_loop_len, out_bin_dx_min
    
    @cython.final
    def DescribeSystem(self):

        nparam_nosym = self.geodim * self._nint * self.nbody
        nparam_tot = self.nparams_incl_o // 2

        out = ""
        out += 'System is composed of:\n'
        out += f'    {self.nbody:d} bodies\n'
        out += f'    {self.nloop:d} independent loops\n'
        out += f'    {self.nint_min:d} integration segments\n'
        out += f'    {self.nsegm:d} independent generating segments\n'
        out += f'    {self.nbin_segm_unique:d} binary interactions between segments\n'
        out += '\n'
        out += f'The number of free parameters is reduced by a factor of {nparam_nosym / nparam_tot}\n'
        out += f'The number of independent interactions is reduced by a factor of {self.nbin_segm_tot  / self.nbin_segm_unique}\n'
        out += f'The number of independent segments is reduced by a factor of {(self.nbody * self.nint_min) / self.nsegm}\n'
        out += '\n'

        return out    

    @cython.cdivision(True)
    @cython.final
    def ComputeCenterOfMass(self, segmpos):

        cdef Py_ssize_t il, ib, iint
        cdef Py_ssize_t isegm, segmbeg, segmend
        cdef np.ndarray[double, ndim=2, mode='c'] pos = np.empty((self.segm_size, self.geodim), dtype=np.float64)
        cdef np.ndarray[double, ndim=2, mode='c'] pos_avg = np.zeros((self.segm_size, self.geodim), dtype=np.float64)

        cdef int size = self.segm_size * self.geodim
        cdef double mul
        cdef double totmass = 0.

        cdef ActionSym Sym

        for ib in range(self.nbody):
            for iint in range(self.nint_min):

                isegm = self._bodysegm[ib, iint]

                Sym = self.intersegm_to_all[ib][iint]

                ibeg = iint * self.segm_size         
                iend = ibeg + self.segm_size
                assert iend <= self._nint

                if Sym.TimeRev > 0:
                    segmbeg = 0
                    segmend = self.segm_size
                else:
                    segmbeg = 1
                    segmend = self.segm_size+1

                Sym.TransformSegment(segmpos[isegm,segmbeg:segmend,:], pos)

                il = self._bodyloop[ib]
                mul = self._loopmass[il]
                totmass += mul

                scipy.linalg.cython_blas.daxpy(&size,&mul,&pos[0,0],&int_one,&pos_avg[0,0],&int_one)

        mul = 1. / (self.segm_size * totmass)

        return mul * np.sum(pos_avg, axis=0)

    @cython.final
    def Compute_periodicity_default_pos(self, double[::1] xo, double[::1] xf):

        assert xo.shape[0] == self.nsegm * self.geodim
        assert xf.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        cdef Py_ssize_t isegm, idim, jdim, i, j

        for isegm in range(self.nsegm):

            if self._PerDefEnd_TimeRev[isegm] > 0:

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = xf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefEnd_SpaceRotPos[isegm,idim,jdim] * xo[j]
                        j += 1

                    i += 1

            else:

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = xf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefEnd_SpaceRotPos[isegm,idim,jdim] * xf[j]
                        j += 1

                    i += 1

        return res
        
    @cython.final
    def Compute_periodicity_default_vel(self, double[::1] vo, double[::1] vf):

        assert vo.shape[0] == self.nsegm * self.geodim
        assert vf.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        cdef Py_ssize_t isegm, idim, jdim, i, j

        for isegm in range(self.nsegm):

            if self._PerDefEnd_TimeRev[isegm] > 0:

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = vf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefEnd_SpaceRotVel[isegm,idim,jdim] * vo[j]
                        j += 1

                    i += 1

            else:

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = vf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefEnd_SpaceRotVel[isegm,idim,jdim] * vf[j]
                        j += 1

                    i += 1

        return res

    @cython.final
    def Compute_initial_constraint_default_pos(self, double[::1] xo):

        assert xo.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        cdef Py_ssize_t isegm, idim, jdim, i, j

        for isegm in range(self.nsegm):

            if self._PerDefBeg_TimeRev[isegm] < 0:

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = xo[i]

                    j = self._PerDefBeg_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefBeg_SpaceRotPos[isegm,idim,jdim] * xo[j]
                        j += 1

                    i += 1

        return res
        
    @cython.final
    def Compute_initial_constraint_default_vel(self, double[::1] vo):

        assert vo.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        cdef Py_ssize_t isegm, idim, jdim, i, j

        for isegm in range(self.nsegm):

            if self._PerDefBeg_TimeRev[isegm] < 0:

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = vo[i]

                    j = self._PerDefBeg_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefBeg_SpaceRotVel[isegm,idim,jdim] * vo[j]
                        j += 1

                    i += 1

        return res

    @cython.final
    def Compute_init_pos_mom(self, double[::1] params_mom_buf):

        cdef Py_ssize_t isegm, idim, i
        cdef double mass

        segmpos = self.params_to_segmpos(params_mom_buf)
        cdef np.ndarray[double, ndim=1, mode='c'] xo = segmpos[:,0,:].copy().reshape(-1)

        segmvel = self.params_to_segmvel(params_mom_buf)
        cdef np.ndarray[double, ndim=1, mode='c'] po = segmvel[:,0,:].copy().reshape(-1)

        for isegm in range(self.nsegm):
            mass = self._loopmass[self._bodyloop[self._intersegm_to_body[isegm]]]
            for idim in range(self.geodim):
                i = isegm*self.geodim + idim
                po[i] *= mass

        return xo, po

    @cython.final
    @cython.cdivision(True)
    def Compute_velocities(self, double t, double[::1] mom_flat):

        assert mom_flat.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        Compute_velocities_vectorized(
            &mom_flat[0]            , &res[0]       ,
            self.nbin_segm_unique   , self.geodim   ,   
            self.nsegm              , 1             ,  
            &self._invsegmmass[0]   , 
        )

        return res

    @cython.final
    @cython.cdivision(True)
    def Compute_grad_velocities(self, double t, double[::1] mom_flat, double[:,::1] grad_mom_flat):

        assert mom_flat.shape[0] == self.nsegm * self.geodim
        cdef Py_ssize_t grad_ndof = grad_mom_flat.shape[1]

        cdef np.ndarray[double, ndim=2, mode='c'] res = np.empty((self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_velocities_vectorized(
            &mom_flat[0]            , &grad_mom_flat[0,0]   , &res[0,0] ,
            self.nbin_segm_unique   , self.geodim           ,   
            self.nsegm              , 1                     , grad_ndof  ,
            &self._invsegmmass[0]   , 
        )

        return res

    @cython.final
    @cython.cdivision(True)
    def Compute_velocities_vectorized(self, double[::1] t, double[:,::1] mom_flat):

        assert mom_flat.shape[1] == self.nsegm * self.geodim

        cdef Py_ssize_t nvec = mom_flat.shape[0]

        cdef np.ndarray[double, ndim=2, mode='c'] res = np.empty((nvec, self.nsegm * self.geodim), dtype=np.float64)

        Compute_velocities_vectorized(
            &mom_flat[0,0]          , &res[0,0]     ,
            self.nbin_segm_unique   , self.geodim   ,   
            self.nsegm              , nvec          ,  
            &self._invsegmmass[0]   , 
        )

        return res
        
    @cython.final
    @cython.cdivision(True)
    def Compute_grad_velocities_vectorized(self, double[::1] t, double[:,::1] mom_flat, double[:,:,::1] grad_mom_flat):

        assert mom_flat.shape[1] == self.nsegm * self.geodim

        cdef Py_ssize_t nvec = mom_flat.shape[0]
        cdef Py_ssize_t grad_ndof = grad_mom_flat.shape[2]

        cdef np.ndarray[double, ndim=3, mode='c'] res = np.empty((nvec, self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_velocities_vectorized(
            &mom_flat[0,0]          , &grad_mom_flat[0,0,0] , &res[0,0,0]   ,
            self.nbin_segm_unique   , self.geodim           ,   
            self.nsegm              , nvec                  , grad_ndof     ,
            &self._invsegmmass[0]   , 
        )

        return res

    @cython.final
    def Compute_forces(self, double t, double[::1] pos_flat):

        assert pos_flat.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        Compute_forces_vectorized(
            &pos_flat[0]                , &res[0]                   ,
            self.nbin_segm_unique       , self.geodim               ,   
            self.nsegm                  , 1                         ,           
            &self._BinSourceSegm[0]     , &self._BinTargetSegm[0]   ,
            &self._BinSpaceRot[0,0,0]   , &self._BinSpaceRotIsId[0] ,
            &self._BinProdChargeSum[0]  ,
            self._inter_law             , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    def Compute_grad_forces(self, double t, double[::1] pos_flat, double[:,::1] dpos_flat):

        assert pos_flat.shape[0] == self.nsegm * self.geodim

        cdef Py_ssize_t grad_ndof = dpos_flat.shape[1]
        cdef np.ndarray[double, ndim=2, mode='c'] res = np.empty((self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_forces_vectorized(
            &pos_flat[0]                , &dpos_flat[0,0]           , &res[0,0] ,
            self.nbin_segm_unique       , self.geodim               ,   
            self.nsegm                  , 1                         , grad_ndof ,
            &self._BinSourceSegm[0]     , &self._BinTargetSegm[0]   ,
            &self._BinSpaceRot[0,0,0]   , &self._BinSpaceRotIsId[0] ,
            &self._BinProdChargeSum[0]  ,
            self._inter_law             , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    def Compute_forces_nosym(self, double t, double[::1] pos_flat):

        assert pos_flat.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        Compute_forces_vectorized_nosym(
            &pos_flat[0]            , &res[0]                   ,
            self.geodim             ,   
            self.nsegm              , 1                         ,           
            &self._segmcharge[0]    ,
            self._inter_law         , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    def Compute_grad_forces_nosym(self, double t, double[::1] pos_flat, double[:,::1] dpos_flat):

        assert pos_flat.shape[0] == self.nsegm * self.geodim

        cdef Py_ssize_t grad_ndof = dpos_flat.shape[1]
        cdef np.ndarray[double, ndim=2, mode='c'] res = np.empty((self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_forces_vectorized_nosym(
            &pos_flat[0]            , &dpos_flat[0,0]           , &res[0,0] ,
            self.geodim             ,   
            self.nsegm              , 1                         , grad_ndof ,
            &self._segmcharge[0]    ,
            self._inter_law         , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    def Compute_forces_vectorized(self, double[::1] t, double[:,::1] pos_flat):

        assert pos_flat.shape[1] == self.nsegm * self.geodim

        cdef Py_ssize_t nvec = pos_flat.shape[0]

        cdef np.ndarray[double, ndim=2, mode='c'] res = np.empty((nvec, self.nsegm * self.geodim), dtype=np.float64)

        Compute_forces_vectorized(
            &pos_flat[0,0]              , &res[0,0]                 ,
            self.nbin_segm_unique       , self.geodim               , 
            self.nsegm                  , nvec                      ,       
            &self._BinSourceSegm[0]     , &self._BinTargetSegm[0]   ,
            &self._BinSpaceRot[0,0,0]   , &self._BinSpaceRotIsId[0] ,
            &self._BinProdChargeSum[0]  ,
            self._inter_law             , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    def Compute_grad_forces_vectorized(self, double[::1] t, double[:,::1] pos_flat, double[:,:,::1] dpos_flat):

        assert pos_flat.shape[1] == self.nsegm * self.geodim

        cdef Py_ssize_t nvec = pos_flat.shape[0]
        cdef Py_ssize_t grad_ndof = dpos_flat.shape[2]
        cdef np.ndarray[double, ndim=3, mode='c'] res = np.empty((nvec, self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_forces_vectorized(
            &pos_flat[0,0]              , &dpos_flat[0,0,0]         , &res[0,0,0]   ,
            self.nbin_segm_unique       , self.geodim               , 
            self.nsegm                  , nvec                      , grad_ndof     ,
            &self._BinSourceSegm[0]     , &self._BinTargetSegm[0]   ,
            &self._BinSpaceRot[0,0,0]   , &self._BinSpaceRotIsId[0] ,
            &self._BinProdChargeSum[0]  ,
            self._inter_law             , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    def Compute_forces_vectorized_nosym(self, double[::1] t, double[:,::1] pos_flat):

        assert pos_flat.shape[1] == self.nsegm * self.geodim

        cdef Py_ssize_t nvec = pos_flat.shape[0]

        cdef np.ndarray[double, ndim=2, mode='c'] res = np.empty((nvec, self.nsegm * self.geodim), dtype=np.float64)

        Compute_forces_vectorized_nosym(
            &pos_flat[0,0]          , &res[0,0]                 ,
            self.geodim             , 
            self.nsegm              , nvec                      ,       
            &self._segmcharge[0]    ,
            self._inter_law         , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    def Compute_grad_forces_vectorized_nosym(self, double[::1] t, double[:,::1] pos_flat, double[:,:,::1] dpos_flat):

        assert pos_flat.shape[1] == self.nsegm * self.geodim

        cdef Py_ssize_t nvec = pos_flat.shape[0]
        cdef Py_ssize_t grad_ndof = dpos_flat.shape[2]

        cdef np.ndarray[double, ndim=3, mode='c'] res = np.empty((nvec, self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_forces_vectorized_nosym(
            &pos_flat[0,0]          , &dpos_flat[0,0,0]         , &res[0,0,0]   ,
            self.geodim             , 
            self.nsegm              , nvec                      , grad_ndof     ,
            &self._segmcharge[0]    ,
            self._inter_law         , self._inter_law_param_ptr ,
        )

        return res

    @cython.final
    @cython.cdivision(True)
    def Get_ODE_def(self, double[::1] params_mom_buf, vector_calls = True, LowLevel = True, NoSymIfPossible = True, grad = False):

        xo, po = self.Compute_init_pos_mom(params_mom_buf)

        dict_res = {
            "t_span" : (0., 1./self.nint_min)   ,
            "x0" : xo                           ,
            "v0" : po                           ,
            "vector_calls" : vector_calls       ,
        }

        NoSymPossible = self.BinSpaceRotIsId.all()
        NoSym = NoSymIfPossible and NoSymPossible

        if LowLevel:

            user_data = self.Get_ODE_params()

            if vector_calls:

                dict_res["fun"] = scipy.LowLevelCallable.from_cython(
                    choreo.cython._NBodySyst                    ,
                    "Compute_velocities_vectorized_user_data"   ,
                    user_data                                   ,
                )
                if grad:
                    dict_res["grad_fun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst                        ,
                        "Compute_grad_velocities_vectorized_user_data"  ,
                        user_data                                       ,
                    )

                if NoSym:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst                    ,
                        "Compute_forces_vectorized_nosym_user_data" ,
                        user_data                                   ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst                            ,
                            "Compute_grad_forces_vectorized_nosym_user_data"    ,
                            user_data                                           ,
                        )

                else:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst                    ,
                        "Compute_forces_vectorized_user_data"       ,
                        user_data                                   ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst                    ,
                            "Compute_grad_forces_vectorized_user_data"  ,
                            user_data                                   ,
                        )

            else:

                dict_res["fun"] = scipy.LowLevelCallable.from_cython(
                    choreo.cython._NBodySyst                    ,
                    "Compute_velocities_user_data"              ,
                    user_data                                   ,
                )
                if grad:
                    dict_res["grad_fun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst                    ,
                        "Compute_grad_velocities_user_data"         ,
                        user_data                                   ,
                    )

                if NoSym:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst                ,
                        "Compute_forces_nosym_user_data"        ,
                        user_data                               ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst                ,
                            "Compute_grad_forces_nosym_user_data"   ,
                            user_data                               ,
                        )

                else:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst                ,
                        "Compute_forces_user_data"              ,
                        user_data                               ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst        ,
                            "Compute_grad_forces_user_data" ,
                            user_data                       ,
                        )

        else:

            if vector_calls:

                dict_res["fun"] = self.Compute_velocities_vectorized
                if grad:
                    dict_res["grad_fun"] = self.Compute_grad_velocities_vectorized

                if NoSym:
                    dict_res["gun"] = self.Compute_forces_vectorized_nosym
                    if grad:
                        dict_res["grad_gun"] = self.Compute_grad_forces_vectorized_nosym

                else:
                    dict_res["gun"] = self.Compute_forces_vectorized
                    if grad:
                        dict_res["grad_gun"] = self.Compute_grad_forces_vectorized

            else:

                dict_res["fun"] = self.Compute_velocities
                if grad:
                    dict_res["grad_fun"] = self.Compute_grad_velocities
                
                if NoSym:
                    dict_res["gun"] = self.Compute_forces_nosym
                    if grad:
                        dict_res["grad_gun"] = self.Compute_grad_forces_nosym

                else:
                    dict_res["gun"] = self.Compute_forces
                    if grad:
                        dict_res["grad_gun"] = self.Compute_grad_forces

        return dict_res

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
    Py_ssize_t nint_min         ,
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

            out_segm_len[isegm] = nint_min * segm_len_val / segm_size

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

            out_segm_len[isegm] = nint_min * segm_len_val / segm_size

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
    double* pos                     , double* forces            ,
    Py_ssize_t nbin                 , Py_ssize_t geodim         ,
    Py_ssize_t nsegm                , Py_ssize_t nvec           ,
    Py_ssize_t* BinSourceSegm       , Py_ssize_t* BinTargetSegm ,
    double* BinSpaceRot             , bint* BinSpaceRotIsId     ,
    double* BinProdChargeSum        ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
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
                    cur_pos  = pos + isegm *geodim
                    for jdim in range(geodim):
                        dx[idim] += matel[0] * cur_pos[0]
                        matel += 1 
                        cur_pos += 1

                    cur_posp += 1

            dx2 = dx[0]*dx[0]
            for idim in range(1,geodim):
                dx2 += dx[idim]*dx[idim]

            inter_law(dx2, pot, inter_law_param_ptr)

            bin_fac = (-4)*BinProdChargeSum[ibin]

            pot[1] *= bin_fac

            for idim in range(geodim):
                df[idim] = pot[1]*dx[idim]

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

            cur_forces = vec_forces + isegmp*geodim

            for idim in range(geodim):
                cur_forces[0] -= df[idim]
                cur_forces += 1
                
    free(dx)
    free(df)

cdef inline void Compute_grad_forces_vectorized(
    double* pos                     , double* dpos              , double* grad_forces   ,
    Py_ssize_t nbin                 , Py_ssize_t geodim         ,
    Py_ssize_t nsegm                , Py_ssize_t nvec           , Py_ssize_t grad_ndof  ,
    Py_ssize_t* BinSourceSegm       , Py_ssize_t* BinTargetSegm ,
    double* BinSpaceRot             , bint* BinSpaceRotIsId     ,
    double* BinProdChargeSum        ,
    inter_law_fun_type inter_law    , void* inter_law_param_ptr ,
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

    cdef double dx2, dxtddx, a ,b
    cdef double bin_fac
    cdef double[3] pot

    cdef double* dx = <double*> malloc(sizeof(double)*geodim)
    cdef double* ddx = <double*> malloc(sizeof(double)*geodim*grad_ndof)
    cdef double* ddf = <double*> malloc(sizeof(double)*geodim*grad_ndof)

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

                    dx[idim] = cur_pos[0]  - cur_posp[0]

                    cur_pos  += 1
                    cur_posp += 1

            else:

                RotMat = BinSpaceRot + ibin * geodim_sq
                cur_posp  = vec_pos  + isegmp * geodim

                for idim in range(geodim):
                    
                    dx[idim]  = -cur_posp[0]
                    ddx[idim] = -cur_dposp[0]
                    cur_pos   = pos + isegm * geodim

                    for jdim in range(geodim):
                    
                        dx[idim] += RotMat[0] * cur_pos[0]
                    
                        RotMat += 1 
                        cur_pos += 1

                    cur_posp += 1

            dx2 = dx[0]*dx[0]
            for idim in range(1,geodim):
                dx2 += dx[idim]*dx[idim]

            inter_law(dx2, pot, inter_law_param_ptr)

            bin_fac = (-4)*BinProdChargeSum[ibin]

            a = pot[1]*bin_fac
            pot[2] *= 2*bin_fac

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

            if BinSpaceRotIsId[ibin]:

                scipy.linalg.cython_blas.daxpy(&dsegm_size,&one_double, ddf, &int_one, grad_forces_loc, &int_one)

            else:

                RotMat = BinSpaceRot + ibin * geodim_sq
                scipy.linalg.cython_blas.dgemm(transn, transt, &grad_ndof_int, &geodim_int, &geodim_int, &one_double, ddf, &grad_ndof_int, RotMat, &geodim_int, &one_double, grad_forces_loc, &grad_ndof_int)

            
            scipy.linalg.cython_blas.daxpy(&dsegm_size,&minusone_double, ddf, &int_one, grad_forces_locp, &int_one)
                    
    free(dx)
    free(ddx)
    free(ddf)

cdef void Compute_forces_user_data(
    double t    , double[::1] pos   , double[::1] forces    , void* user_data   ,
) noexcept nogil:

    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_forces_vectorized(
        &pos[0]                         , &forces[0]                        ,
        ODE_params.nbin                 , ODE_params.geodim                 , 
        ODE_params.nsegm                , 1                                 ,
        ODE_params.BinSourceSegm_ptr    , ODE_params.BinTargetSegm_ptr      ,
        ODE_params.BinSpaceRot_ptr      , ODE_params.BinSpaceRotIsId_ptr    ,
        ODE_params.BinProdChargeSum_ptr ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
    )

cdef void Compute_grad_forces_user_data(
    double t    , double[::1] pos   , double[:,::1] dpos   , double[:,::1] grad_forces  , void* user_data   ,
) noexcept nogil:

    cdef Py_ssize_t grad_ndof = dpos.shape[1]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_grad_forces_vectorized(
        &pos[0]                         , &dpos[0,0]                        , &grad_forces[0,0] ,
        ODE_params.nbin                 , ODE_params.geodim                 , 
        ODE_params.nsegm                , 1                                 , grad_ndof         ,
        ODE_params.BinSourceSegm_ptr    , ODE_params.BinTargetSegm_ptr      ,
        ODE_params.BinSpaceRot_ptr      , ODE_params.BinSpaceRotIsId_ptr    ,
        ODE_params.BinProdChargeSum_ptr ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
    )

cdef void Compute_forces_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,::1] all_forces, void* user_data ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_pos.shape[0]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_forces_vectorized(
        &all_pos[0,0]                   , &all_forces[0,0]                  ,
        ODE_params.nbin                 , ODE_params.geodim                 ,
        ODE_params.nsegm                , nvec                              ,
        ODE_params.BinSourceSegm_ptr    , ODE_params.BinTargetSegm_ptr      ,
        ODE_params.BinSpaceRot_ptr      , ODE_params.BinSpaceRotIsId_ptr    ,
        ODE_params.BinProdChargeSum_ptr ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
    )

cdef void Compute_grad_forces_vectorized_user_data(
    double[::1] all_t   , double[:,::1] all_pos , double[:,:,::1] all_dpos  , double[:,:,::1] all_grad_forces   , void* user_data ,
) noexcept nogil:

    cdef Py_ssize_t nvec = all_pos.shape[0]
    cdef Py_ssize_t grad_ndof = all_dpos.shape[2]
    cdef ODE_params_t* ODE_params_ptr = <ODE_params_t*> user_data
    cdef ODE_params_t ODE_params = ODE_params_ptr[0]

    Compute_grad_forces_vectorized(
        &all_pos[0,0]                   , &all_dpos[0,0,0]                  , &all_grad_forces[0,0,0]   ,
        ODE_params.nbin                 , ODE_params.geodim                 ,
        ODE_params.nsegm                , nvec                              , grad_ndof                 ,
        ODE_params.BinSourceSegm_ptr    , ODE_params.BinTargetSegm_ptr      ,
        ODE_params.BinSpaceRot_ptr      , ODE_params.BinSpaceRotIsId_ptr    ,
        ODE_params.BinProdChargeSum_ptr ,
        ODE_params.inter_law            , ODE_params.inter_law_param_ptr    ,
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
