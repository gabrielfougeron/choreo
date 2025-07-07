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

import choreo.scipy_plus.cython.misc
cimport scipy.linalg.cython_blas
from choreo.scipy_plus.cython.blas_consts cimport *
from choreo.scipy_plus.cython.ccallback cimport ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS, ccallback_signature_t

from choreo.scipy_plus.cython.kepler cimport kepler
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
    import matplotlib
    import matplotlib.animation
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

# available, use_id, package_name
All_fft_backends = {
    "scipy" : (True                 , USE_SCIPY_FFT , "scipy"   ),
    "mkl"   : (MKL_FFT_AVAILABLE    , USE_MKL_FFT   , "mkl_fft" ),
    "ducc"  : (DUCC_FFT_AVAILABLE   , USE_DUCC_FFT  , "ducc0"   ),
    "fftw"  : (PYFFTW_AVAILABLE     , USE_FFTW_FFT  , "pyfftw"  ),
}

from choreo.cython._NBodySyst_ann cimport *

shortcut_name = {
    GENERAL_SYM : "general_sym" ,
    RFFT        : "rfft"        ,
}

cdef ccallback_signature_t signatures[2]

signatures[0].signature = b"void (double, double *, void *)"
signatures[0].value = 0
signatures[1].signature = NULL

default_GUI_colors = [
	"#50ce4d", # Moderate Lime Green
	"#ff7006", # Vivid Orange
	"#a253c4", # Moderate Violet
	"#ef1010", # Vivid Red
	"#25b5bc", # Strong Cyan
	"#E86A96", # Soft Pink
	"#edc832", # Bright Yellow
	"#ad6530", # Dark Orange [Brown tone]
	"#00773f", # Dark cyan - lime green 
	"#d6d6d6", # Light gray
]

FallbackTrailColor = "#d5d5d5"
bgColor = "#F1F1F1"

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

        backend_prop = All_fft_backends.get(backend)

        if backend_prop is None:

            err_message = 'Invalid FFT backend. Here is an overview of possible options:\n'
            available_dict = {True:"available",False:"unavailable"}

            for name, (available, use_id, package_name) in All_fft_backends.items():
                err_message += f'Backend "{name}" from package {package_name} is {available_dict[available]}.\n'

            raise ValueError(err_message)
        
        else:

            available, use_id, package_name = backend_prop
            if available:
                self._fft_backend = use_id
            else:
                raise ValueError(f"The package {package_name} could not be loaded. Please check your local install.")

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

    cdef double[::1] _segmmass
    @property
    def segmmass(self):
        return np.asarray(self._segmmass)

    cdef double[::1] _invsegmmass
    @property
    def invsegmmass(self):
        return np.asarray(self._invsegmmass)

    cdef double[::1] _segmcharge
    @property
    def segmcharge(self):
        return np.asarray(self._segmcharge)

    cdef double[::1] _BinProdChargeSumSource_ODE
    @property
    def BinProdChargeSumSource_ODE(self):
        return np.asarray(self._BinProdChargeSumSource_ODE)

    cdef double[::1] _BinProdChargeSumTarget_ODE
    @property
    def BinProdChargeSumTarget_ODE(self):
        return np.asarray(self._BinProdChargeSumTarget_ODE)

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

    cdef readonly Py_ssize_t TimeRev

    cdef double[:,:,::1] _PerDefEnd_SpaceRotPos
    @property
    def PerDefEnd_SpaceRotPos(self):
        return np.asarray(self._PerDefEnd_SpaceRotPos)

    cdef double[:,:,::1] _PerDefEnd_SpaceRotVel
    @property
    def PerDefEnd_SpaceRotVel(self):
        return np.asarray(self._PerDefEnd_SpaceRotVel)

    cdef double[:,:,::1] _CoMMat
    @property
    def CoMMat(self):
        return np.asarray(self._CoMMat)

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

    cdef double[:,::1] _ODEinitparams_basis_pos
    @property
    def ODEinitparams_basis_pos(self):
        return np.asarray(self._ODEinitparams_basis_pos)

    cdef double[:,::1] _ODEinitparams_basis_mom
    @property
    def ODEinitparams_basis_mom(self):
        return np.asarray(self._ODEinitparams_basis_mom)

    cdef readonly Py_ssize_t n_ODEinitparams_pos
    cdef readonly Py_ssize_t n_ODEinitparams_mom
    cdef readonly Py_ssize_t n_ODEinitparams

    cdef double[:,::1] _ODEperdef_eqproj_pos
    @property
    def ODEperdef_eqproj_pos(self):
        return np.asarray(self._ODEperdef_eqproj_pos)

    cdef double[:,::1] _ODEperdef_eqproj_mom
    @property
    def ODEperdef_eqproj_mom(self):
        return np.asarray(self._ODEperdef_eqproj_mom)

    cdef readonly Py_ssize_t n_ODEperdef_eqproj_pos
    cdef readonly Py_ssize_t n_ODEperdef_eqproj_mom
    cdef readonly Py_ssize_t n_ODEperdef_eqproj

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
            npr = self._nint // (2*self._ncoeff_min_loop[il])
            
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
    @property
    def segmpos(self):
        return np.asarray(self._segmpos)

    @segmpos.setter
    @cython.cdivision(True)
    @cython.final
    def segmpos(self, double[:,:,::1] segmpos_in):
        self._segmpos = segmpos_in

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

        BinarySegm = FindAllBinarySegments(self.intersegm_to_all, nbody, self.nsegm, self.nint_min, self._intersegm_to_body, self._bodysegm, bodycharge)

        self.nbin_segm_tot, self.nbin_segm_unique = CountSegmentBinaryInteractions(BinarySegm, self.nsegm)

        self._BinSourceSegm, self._BinTargetSegm, self._BinSpaceRot, self._BinProdChargeSum, self._BinProdChargeSumSource_ODE, self._BinProdChargeSumTarget_ODE = ReorganizeBinarySegments(BinarySegm)

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

        self.SetConvenienceArrays()
        self.SetODEArrays()
        self.Update_ODE_params()

        self.Find_ODE_params_basis()

        if MKL_FFT_AVAILABLE:
            self.fft_backend = "mkl"
        elif DUCC_FFT_AVAILABLE:
            self.fft_backend = "ducc"
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
        self._ODE_params.BinProdChargeSumSource_ptr = &self._BinProdChargeSumSource_ODE[0]
        self._ODE_params.BinProdChargeSumTarget_ptr = &self._BinProdChargeSumTarget_ODE[0]

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
        self._intersegm_to_body = np.empty((self.nsegm), dtype = np.intp)
        self._intersegm_to_iint = np.empty((self.nsegm), dtype = np.intp)

        assigned_segms = set()

        for iint in range(self.nint_min):
            for ib in range(self.nbody):

                isegm = self._bodysegm[ib,iint]

                if not(isegm in assigned_segms):
                    
                    self._intersegm_to_body[isegm] = ib
                    self._intersegm_to_iint[isegm] = iint
                    assigned_segms.add(isegm)

        #Every interaction happens at iint == 0
        if not (np.asarray(self._intersegm_to_iint) == 0).all():
            raise ValueError("Catastrophic failure: interacting segments are not all at initial positions")

    @cython.final
    def GatherInterSym(self):

        cdef Py_ssize_t isegm
        cdef ActionSym Sym

        self._InterTimeRev = np.empty((self.nsegm), dtype=np.intp)
        self._InterSpaceRotPos = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        self._InterSpaceRotPosIsId = np.empty((self.nsegm), dtype=np.intc)
        self._InterSpaceRotVel = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        self._InterSpaceRotVelIsId = np.empty((self.nsegm), dtype=np.intc)

        for isegm in range(self.nsegm):

            ib = self._gensegm_to_body[isegm]
            iint = self._gensegm_to_iint[isegm]

            Sym = self.intersegm_to_all[ib][iint]
            
            self._InterTimeRev[isegm] = Sym.TimeRev
            self._InterSpaceRotPos[isegm,:,:] = Sym._SpaceRot[:,:]
            self._InterSpaceRotPosIsId[isegm] = Sym.IsIdentityRot()

            Sym = Sym.TimeDerivative()
            self._InterSpaceRotVel[isegm,:,:] = Sym._SpaceRot[:,:]
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

    @cython.final
    def SetConvenienceArrays(self):

        self.TimeRev = self.intersegm_to_all[0][1%self.nint_min].TimeRev

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
        self._PerDefBeg_SpaceRotPos = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        self._PerDefBeg_SpaceRotVel = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)

        for isegm in range(self.nsegm):

            iint_uneven = self.nint_min - 1

            ib = self._intersegm_to_body[isegm]

            self._PerDefBeg_Isegm[isegm] = self._bodysegm[ib, iint_uneven]

            Sym = self.intersegm_to_all[ib][iint_uneven]
            assert self.TimeRev == Sym.TimeRev
            self._PerDefBeg_SpaceRotPos[isegm,:,:] = Sym._SpaceRot[:,:]

            Sym = Sym.TimeDerivative()
            self._PerDefBeg_SpaceRotVel[isegm,:,:] = Sym._SpaceRot[:,:]

        self._PerDefEnd_Isegm = np.empty((self.nsegm), dtype=np.intp)
        self._PerDefEnd_SpaceRotPos = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        self._PerDefEnd_SpaceRotVel = np.empty((self.nsegm, self.geodim, self.geodim), dtype=np.float64)

        for isegm in range(self.nsegm):

            iint_uneven = 1 % self.nint_min

            ib = self._intersegm_to_body[isegm]

            self._PerDefEnd_Isegm[isegm] = self._bodysegm[ib, iint_uneven]

            Sym = self.intersegm_to_all[ib][iint_uneven]
            assert self.TimeRev == Sym.TimeRev
            self._PerDefEnd_SpaceRotPos[isegm,:,:] = Sym._SpaceRot[:,:]

            Sym = Sym.TimeDerivative()
            self._PerDefEnd_SpaceRotVel[isegm,:,:] = Sym._SpaceRot[:,:]

        cdef double mass, totmass
        totmass = 0.

        cdef int size
        cdef Py_ssize_t iint

        self._CoMMat = np.zeros((self.geodim, self.nsegm, self.geodim), dtype=np.float64)

        size = self.geodim * self.geodim

        for ib in range(self.nbody):

            il = self._bodyloop[ib]
            mul = self._loopmass[il]
            totmass += mul

            isegm = self._bodysegm[ib, 0]
            Sym = self.intersegm_to_all[ib][0]

            assert Sym.TimeRev == 1

            for idim in range(self.geodim):
                for jdim in range(self.geodim):
                   self._CoMMat[idim, isegm, jdim] += mul * Sym._SpaceRot[idim, jdim]

        size = self.nsegm * self.geodim * self.geodim
        mass = 1./totmass
        scipy.linalg.cython_blas.dscal(&size,&mass,&self._CoMMat[0,0,0],&int_one)

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
    
    @cython.final
    def SetODEArrays(self):

        cdef Py_ssize_t isegm, iint, ib

        self._segmmass = np.empty((self.nsegm), dtype=np.float64)
        self._invsegmmass = np.empty((self.nsegm), dtype=np.float64)
        self._segmcharge = np.empty((self.nsegm), dtype=np.float64)
        for isegm in range(self.nsegm):
            self._segmmass[isegm] = self._loopmass[self._bodyloop[self._intersegm_to_body[isegm]]]
            self._invsegmmass[isegm] = 1. / (self._loopmass[self._bodyloop[self._intersegm_to_body[isegm]]])
            self._segmcharge[isegm] = self._loopcharge[self._bodyloop[self._intersegm_to_body[isegm]]]

    @cython.final
    def Find_ODE_params_basis(self, bint MomCons = True):

        cdef Py_ssize_t ncstr, icstr, isegm, jsegm, idim, jdim, iparam
        cdef double[:,:,:,::1] cstr_mat
        cdef double eps = 1e-12

        ncstr = 0

        if MomCons:
            ncstr += 1

        if self.TimeRev < 0:
            ncstr += self.nsegm

        cstr_mat = np.zeros((ncstr, self.geodim, self.nsegm, self.geodim), dtype=np.float64)
        icstr = -1

        if MomCons:
            icstr += 1
            for idim in range(self.geodim):
                for isegm in range(self.nsegm):
                    for jdim in range(self.geodim):
                        cstr_mat[icstr, idim, isegm, jdim] = self._CoMMat[idim, isegm, jdim]

        if self.TimeRev < 0:

            for isegm in range(self.nsegm):

                icstr += 1
                jsegm = self._PerDefBeg_Isegm[isegm]

                for idim in range(self.geodim):
                    cstr_mat[icstr, idim, isegm, idim] += 1.
                    for jdim in range(self.geodim):
                        cstr_mat[icstr, idim, jsegm, jdim] -= self._PerDefBeg_SpaceRotPos[isegm, idim, jdim]

        cstr_mat_reshape = np.asarray(cstr_mat).reshape((ncstr*self.geodim, self.nsegm*self.geodim))
        choreo.scipy_plus.cython.misc.proj_to_zero(cstr_mat_reshape, eps=eps)
        NullSpace_pos = choreo.scipy_plus.linalg.null_space(cstr_mat_reshape)
        choreo.scipy_plus.cython.misc.proj_to_zero(NullSpace_pos, eps=eps)

        self._ODEinitparams_basis_pos = NullSpace_pos

        cstr_mat = np.zeros((ncstr, self.geodim, self.nsegm, self.geodim), dtype=np.float64)
        icstr = -1

        if MomCons:
            icstr += 1
            for idim in range(self.geodim):
                for isegm in range(self.nsegm):
                    for jdim in range(self.geodim):
                        cstr_mat[icstr, idim, isegm, jdim] = self._CoMMat[idim, isegm, jdim]

        if self.TimeRev < 0:

            for isegm in range(self.nsegm):

                icstr += 1
                jsegm = self._PerDefBeg_Isegm[isegm]

                for idim in range(self.geodim):
                    cstr_mat[icstr, idim, isegm, idim] += 1.
                    for jdim in range(self.geodim):
                        cstr_mat[icstr, idim, jsegm, jdim] -= self._PerDefBeg_SpaceRotVel[isegm, idim, jdim]

        cstr_mat_reshape = np.asarray(cstr_mat).reshape((ncstr*self.geodim, self.nsegm*self.geodim))
        choreo.scipy_plus.cython.misc.proj_to_zero(cstr_mat_reshape, eps=eps)
        NullSpace_mom = choreo.scipy_plus.linalg.null_space(cstr_mat_reshape)
        choreo.scipy_plus.cython.misc.proj_to_zero(NullSpace_mom, eps=eps)
        
        self._ODEinitparams_basis_mom = NullSpace_mom

        self.n_ODEinitparams_pos = self._ODEinitparams_basis_pos.shape[1]
        self.n_ODEinitparams_mom = self._ODEinitparams_basis_mom.shape[1]
        self.n_ODEinitparams = self.n_ODEinitparams_pos + self.n_ODEinitparams_mom

        if MomCons:

            ncstr = 1
            cstr_mat = np.zeros((ncstr, self.geodim, self.nsegm, self.geodim), dtype=np.float64)

            # Use dcopy ?
            for idim in range(self.geodim):
                for isegm in range(self.nsegm):
                    for jdim in range(self.geodim):
                        cstr_mat[0, idim, isegm, jdim] += self._CoMMat[idim, isegm, jdim]

            cstr_mat_reshape = np.asarray(cstr_mat).reshape((ncstr*self.geodim, self.nsegm*self.geodim))
            choreo.scipy_plus.cython.misc.proj_to_zero(cstr_mat_reshape, eps=eps)
            NullSpace_MomCons = choreo.scipy_plus.linalg.null_space(cstr_mat_reshape)
            choreo.scipy_plus.cython.misc.proj_to_zero(NullSpace_MomCons, eps=eps)

            # dgemm
            MomCons_proj = np.matmul(NullSpace_MomCons, NullSpace_MomCons.T)


        ncstr = self.nsegm
        cstr_mat = np.zeros((ncstr, self.geodim, self.nsegm, self.geodim), dtype=np.float64)
        icstr = -1

        if self.TimeRev > 0:

            for isegm in range(self.nsegm):
                icstr += 1

                for idim in range(self.geodim):
                    cstr_mat[icstr, idim, isegm, idim] += 1.

        else:

            for isegm in range(self.nsegm):

                icstr += 1
                jsegm = self._PerDefEnd_Isegm[isegm]

                for idim in range(self.geodim):
                    cstr_mat[icstr, idim, isegm, idim] += 1.
                    for jdim in range(self.geodim):
                        cstr_mat[icstr, idim, jsegm, jdim] -= self._PerDefEnd_SpaceRotPos[jsegm, idim, jdim]


        cstr_mat_reshape = np.asarray(cstr_mat).reshape((ncstr*self.geodim, self.nsegm*self.geodim))
        if MomCons:
            cstr_mat_reshape = np.matmul(cstr_mat_reshape, MomCons_proj)

        choreo.scipy_plus.cython.misc.proj_to_zero(cstr_mat_reshape, eps=eps)
        ImageSpace_pos = np.ascontiguousarray(scipy.linalg.orth(cstr_mat_reshape, eps).T)
        choreo.scipy_plus.cython.misc.proj_to_zero(ImageSpace_pos, eps=eps)

        self._ODEperdef_eqproj_pos = ImageSpace_pos


        ncstr = self.nsegm
        cstr_mat = np.zeros((ncstr, self.geodim, self.nsegm, self.geodim), dtype=np.float64)
        icstr = -1

        if self.TimeRev > 0:

            for isegm in range(self.nsegm):
                icstr += 1

                for idim in range(self.geodim):
                    cstr_mat[icstr, idim, isegm, idim] += 1.

        else:

            for isegm in range(self.nsegm):

                icstr += 1
                jsegm = self._PerDefEnd_Isegm[isegm]

                for idim in range(self.geodim):
                    cstr_mat[icstr, idim, isegm, idim] += 1.
                    for jdim in range(self.geodim):
                        cstr_mat[icstr, idim, jsegm, jdim] -= self._PerDefEnd_SpaceRotVel[jsegm, idim, jdim]


        cstr_mat_reshape = np.asarray(cstr_mat).reshape((ncstr*self.geodim, self.nsegm*self.geodim))
        if MomCons:
            cstr_mat_reshape = np.matmul(cstr_mat_reshape, MomCons_proj)

        choreo.scipy_plus.cython.misc.proj_to_zero(cstr_mat_reshape, eps=eps)
        ImageSpace_mom = np.ascontiguousarray(scipy.linalg.orth(cstr_mat_reshape, eps).T)
        choreo.scipy_plus.cython.misc.proj_to_zero(ImageSpace_mom, eps=eps)

        self._ODEperdef_eqproj_mom = ImageSpace_mom

        self.n_ODEperdef_eqproj_pos = self._ODEperdef_eqproj_pos.shape[0]
        self.n_ODEperdef_eqproj_mom = self._ODEperdef_eqproj_mom.shape[0]
        self.n_ODEperdef_eqproj = self.n_ODEperdef_eqproj_pos + self.n_ODEperdef_eqproj_mom

        assert self.n_ODEperdef_eqproj == self.n_ODEinitparams

    @cython.final
    def ODE_params_to_initposmom(self, double[::1] ODEinitparams):

        cdef int n = self.nsegm*self.geodim
        cdef int m

        assert ODEinitparams.shape[0] == self.n_ODEinitparams

        cdef np.ndarray[double, ndim=1, mode='c'] xo = np.empty((n), dtype=np.float64)
        cdef np.ndarray[double, ndim=1, mode='c'] vo = np.empty((n), dtype=np.float64)

        m = self.n_ODEinitparams_pos
        scipy.linalg.cython_blas.dgemv(transt,&m,&n,&one_double,&self._ODEinitparams_basis_pos[0,0],&m,&ODEinitparams[0],&int_one,&zero_double,&xo[0],&int_one)

        m = self.n_ODEinitparams_mom
        scipy.linalg.cython_blas.dgemv(transt,&m,&n,&one_double,&self._ODEinitparams_basis_mom[0,0],&m,&ODEinitparams[self.n_ODEinitparams_pos],&int_one,&zero_double,&vo[0],&int_one)

        return (xo, vo)

    @cython.final
    def initposmom_to_ODE_params(self, double[::1] xo, double[::1] vo):
    
        cdef int n = self.nsegm*self.geodim
        cdef int m

        assert xo.shape[0] == n
        assert vo.shape[0] == n

        cdef np.ndarray[double, ndim=1, mode='c'] ODEinitparams = np.empty((self.n_ODEinitparams), dtype=np.float64)

        m = self.n_ODEinitparams_pos
        scipy.linalg.cython_blas.dgemv(transn,&m,&n,&one_double,&self._ODEinitparams_basis_pos[0,0],&m,&xo[0],&int_one,&zero_double,&ODEinitparams[0],&int_one)

        m = self.n_ODEinitparams_mom
        scipy.linalg.cython_blas.dgemv(transn,&m,&n,&one_double,&self._ODEinitparams_basis_mom[0,0],&m,&vo[0],&int_one,&zero_double,&ODEinitparams[self.n_ODEinitparams_pos],&int_one)

        return ODEinitparams

    @cython.final
    def endposmom_to_perdef(self, double[::1] xo, double[::1] vo, double[::1] xf, double[::1] vf):
    
        cdef int n
        cdef int m = self.nsegm*self.geodim

        assert xo.shape[0] == m
        assert vo.shape[0] == m
        assert xf.shape[0] == m
        assert vf.shape[0] == m

        cdef np.ndarray[double, ndim=1, mode='c'] ODEperdef = np.empty((self.n_ODEperdef_eqproj), dtype=np.float64)
        cdef np.ndarray[double, ndim=1, mode='c'] buf

        if self.TimeRev > 0:

            buf = np.empty((self.nsegm * self.geodim), dtype=np.float64)

            for isegm in range(self.nsegm):

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    buf[i] = xf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        buf[i] -= self._PerDefEnd_SpaceRotPos[isegm,idim,jdim] * xo[j]
                        j += 1

                    i += 1

            n = self.n_ODEperdef_eqproj_pos
            scipy.linalg.cython_blas.dgemv(transt,&m,&n,&one_double,&self._ODEperdef_eqproj_pos[0,0],&m,&buf[0],&int_one,&zero_double,&ODEperdef[0],&int_one)

            for isegm in range(self.nsegm):

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    buf[i] = vf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        buf[i] -= self._PerDefEnd_SpaceRotVel[isegm,idim,jdim] * vo[j]
                        j += 1

                    i += 1

            n = self.n_ODEperdef_eqproj_mom
            scipy.linalg.cython_blas.dgemv(transt,&m,&n,&one_double,&self._ODEperdef_eqproj_mom[0,0],&m,&buf[0],&int_one,&zero_double,&ODEperdef[self.n_ODEperdef_eqproj_pos],&int_one)

        else:

            n = self.n_ODEperdef_eqproj_pos
            scipy.linalg.cython_blas.dgemv(transt,&m,&n,&one_double,&self._ODEperdef_eqproj_pos[0,0],&m,&xf[0],&int_one,&zero_double,&ODEperdef[0],&int_one)

            n = self.n_ODEperdef_eqproj_mom
            scipy.linalg.cython_blas.dgemv(transt,&m,&n,&one_double,&self._ODEperdef_eqproj_mom[0,0],&m,&vf[0],&int_one,&zero_double,&ODEperdef[self.n_ODEperdef_eqproj_pos],&int_one)

        return ODEperdef

    @cython.final
    @staticmethod
    def all_coeffs_pos_to_vel_inplace(double complex[:,:,::1] all_coeffs):

        cdef int nloop = all_coeffs.shape[0]
        cdef int ncoeffs = all_coeffs.shape[1]
        cdef int geodim = all_coeffs.shape[2]

        cdef Py_ssize_t il, k, idim

        cdef double complex fac = 1j*ctwopi
        cdef double complex mul

        for il in range(nloop):
            for k in range(ncoeffs):
                mul = fac*k
                for idim in range(geodim):
                    # all_coeffs[il,k,idim] *= mul # Causes weird Cython error on Windows
                    all_coeffs[il,k,idim] = all_coeffs[il,k,idim] * mul

    @cython.final
    @staticmethod
    def Get_segmpos_minmax(
        double[:,:,::1] segmpos ,
    ):

        cdef Py_ssize_t nsegm = segmpos.shape[0]
        cdef Py_ssize_t segm_store = segmpos.shape[1]
        cdef Py_ssize_t geodim = segmpos.shape[2]

        segmpos_minmax_np = np.empty((nsegm,2,geodim), dtype=np.float64)
        cdef double[:,:,::1] segmpos_minmax = segmpos_minmax_np

        cdef Py_ssize_t iint, idim, isegm
        cdef double* mi = <double*> malloc(sizeof(double)*geodim)
        cdef double* ma = <double*> malloc(sizeof(double)*geodim)

        cdef double* segmpos_ptr = &segmpos[0,0,0]
        cdef double val

        for isegm in range(nsegm):

            for idim in range(geodim):

                mi[idim] =  DBL_MAX
                ma[idim] = -DBL_MAX

            for iint in range(segm_store):
                for idim in range(geodim):

                    val = segmpos_ptr[0]

                    mi[idim] = min(mi[idim], val)
                    ma[idim] = max(ma[idim], val)

                    segmpos_ptr += 1
        
            for idim in range(geodim):

                segmpos_minmax[isegm,0,idim] = mi[idim]
                segmpos_minmax[isegm,1,idim] = ma[idim]

        free(mi)
        free(ma)

        return segmpos_minmax_np

    @cython.final
    def copy_nosym(self):

        cdef Py_ssize_t ib
        cdef double[::1] bodymass = np.empty((self.nbody), dtype=np.float64)
        cdef double[::1] bodycharge = np.empty((self.nbody), dtype=np.float64)

        for ib in range(self.nbody):
            bodymass[ib] = self._loopmass[self._bodyloop[ib]]
            bodycharge[ib] = self._loopcharge[self._bodyloop[ib]]

        NBS = NBodySyst(
            geodim = self.geodim                                ,
            nbody = self.nbody                                  ,
            bodymass = bodymass                                 ,
            bodycharge = bodycharge                             ,
            Sym_list = []                                       ,
            inter_law_str = self.inter_law_str                  ,
            inter_law_param_dict = self.inter_law_param_dict    ,
        )

        NBS.nint = self.nint

        return NBS

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
        # InterSegmTimeRev = []

        for ib in range(self.nbody):
            InterSegmSpaceRot_b = []
            # InterSegmTimeRev_b = []
            for iint in range(self.nint_min):

                Sym = self.intersegm_to_all[ib][iint]

                InterSegmSpaceRot_b.append(Sym.SpaceRot.tolist())
                # InterSegmTimeRev_b.append(Sym.TimeRev)

            InterSegmSpaceRot.append(InterSegmSpaceRot_b)
            # InterSegmTimeRev.append(InterSegmTimeRev_b)
        
        Info_dict["InterSegmSpaceRot"] = InterSegmSpaceRot
        Info_dict["TimeRev"] = self.TimeRev

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
    @cython.cdivision(True)
    @staticmethod
    def KeplerEllipse(Py_ssize_t nbody = 2, Py_ssize_t nint_fac = 128, double mass = 1., double eccentricity = 0.):

        assert nbody > 1
        assert nint_fac > 0
        assert mass > 0.
        assert eccentricity >= 0.
        assert eccentricity < 1.

        geodim = 2
        bodymass = np.full(nbody, mass, dtype=np.float64)
        bodycharge = bodymass.copy()

        cdef Py_ssize_t ib, iint
        cdef Py_ssize_t[::1] BodyPerm = np.empty(nbody, dtype=np.intp)

        for ib in range(nbody-1):
            BodyPerm[ib] = ib+1

        BodyPerm[nbody-1] = 0

        cdef double angle = ctwopi / nbody 
        cdef double cos_angle = ccos(angle)
        cdef double sin_angle = csin(angle)
        cdef double[:,::1] SpaceRot = np.array([[cos_angle, sin_angle],[-sin_angle, cos_angle]], dtype=np.float64)

        Sym_1 = ActionSym(
            BodyPerm.copy() ,
            SpaceRot.copy() ,
            1               ,
            0               ,
            1               ,
        )

        BodyPerm[0] = 0
        for ib in range(1,nbody):
            BodyPerm[ib] = nbody-ib

        SpaceRot = np.array([[1., 0.],[0., -1.]], dtype=np.float64)

        Sym_2 = ActionSym(
            BodyPerm.copy() ,
            SpaceRot.copy() ,
            -1              ,
            0               ,
            1               ,
        )

        Sym_list = [Sym_1, Sym_2]

        NBS = NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list)
        NBS.nint_fac = nint_fac

        assert NBS.nsegm == 1
        assert NBS.geodim == 2

        dict_res = NBS.Get_ODE_def()

        cdef np.ndarray[double, ndim=3, mode='c'] segmpos_np = np.empty((NBS.nsegm, NBS.segm_store, NBS.geodim), dtype=np.float64)
        cdef double[:,:,::1] segmpos = segmpos_np
        dict_res["segmpos"] = segmpos_np

        cdef np.ndarray[double, ndim=3, mode='c'] segmmom_np = np.empty((NBS.nsegm, NBS.segm_store, NBS.geodim), dtype=np.float64)
        cdef double[:,:,::1] segmmom = segmmom_np
        dict_res["segmmom"] = segmmom_np

        cdef double r, p, fac, a
        cdef double eccentric_anomaly, cos_true_anomaly, sin_true_anomaly, mean_anomaly 
        cdef double dcos_true_anomaly, dsin_true_anomaly

        fac = 0.
        for ib in range(1,nbody):
            angle = cpi * ib / nbody
            fac += 1./csin(angle)

        fac /= ceightpisq

        p = ((1 - eccentricity) * (1 + eccentricity)) * (fac*mass) ** (1./3)

        for iint in range(NBS.segm_store):

            mean_anomaly = (cpi * iint) / NBS.segm_size

            eccentric_anomaly, cos_true_anomaly, sin_true_anomaly, dcos_true_anomaly, dsin_true_anomaly = kepler(mean_anomaly, eccentricity)

            fac = 1. / (1. + eccentricity * cos_true_anomaly)

            r = p * fac

            segmpos[0, iint, 0] = r * cos_true_anomaly
            segmpos[0, iint, 1] = r * sin_true_anomaly

            r *= ctwopi
            dcos_true_anomaly *= fac

            segmmom[0, iint, 0] =  r * dcos_true_anomaly
            segmmom[0, iint, 1] =  r * ( ( - eccentricity * sin_true_anomaly) * dcos_true_anomaly + dsin_true_anomaly)

        NBS.inplace_segmvel_to_segmmom(segmmom)

        dict_res["reg_xo"] = np.ascontiguousarray(segmpos_np.swapaxes(0, 1).reshape(NBS.segm_store,-1))
        dict_res["reg_vo"] = np.ascontiguousarray(segmmom_np.swapaxes(0, 1).reshape(NBS.segm_store,-1))

        return NBS, dict_res

    @cython.final
    def GetKrylovJacobian(self, Use_exact_Jacobian=True, jac_options_kw={}):

        if (Use_exact_Jacobian):

            jacobian = scipy.optimize.nonlin.KrylovJacobian(**jac_options_kw)

            def matvec(self,v):                
                return self.NBS.segmpos_dparams_to_action_hess(self.segmpos, v)

            jacobian.matvec = types.MethodType(matvec, jacobian)
            jacobian.rmatvec = types.MethodType(matvec, jacobian)

        else: 

            jacobian = scipy.optimize.nonlin.KrylovJacobian(**jac_options_kw)
    
        def update(self, x, f):
            self.segmpos = self.NBS.segmpos.copy()
            scipy.optimize.nonlin.KrylovJacobian.update(self, x, f)

        def setup(self, x, f, func):
            self.segmpos = self.NBS.segmpos.copy()
            scipy.optimize.nonlin.KrylovJacobian.setup(self, x, f, func)

        jacobian.update = types.MethodType(update, jacobian)
        jacobian.setup = types.MethodType(setup, jacobian)

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
    def plot_segmpos_2D(self, segmpos, filename, fig_size=(10,10), dpi=100, color=None, color_list=default_GUI_colors, xlim=None, extend=0.03, Mass_Scale=True, trail_width=3.):
        """
        Plots 2D trajectories with one color per body and saves image in file
        """

        cdef Py_ssize_t ib, iint 
        cdef Py_ssize_t il, loop_id
        cdef double mass, line_width, point_size

        assert self.geodim == 2
        assert segmpos.shape[1] == self.segm_store
        
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
        fig = plt.figure(facecolor=bgColor)
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

                    line_width = trail_width

                    if Mass_Scale:
                        mass = self._loopmass[self._bodyloop[ib]]
                        line_width = line_width * mass

                    plt.plot(pos[:,0], pos[:,1], color=current_color, antialiased=True, zorder=-iplt, linewidth=line_width)

        ax.axis('off')
        ax.set_xlim([xinf, xsup])
        ax.set_ylim([yinf, ysup ])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        plt.savefig(filename)
        
        plt.close()

    @cython.final
    def plot_all_2D_anim(self, allpos, filename, fig_size=(10,10), dpi=100, color="body", color_list=default_GUI_colors, xlim=None, extend=0.03, fps=60., bint Mass_Scale=True, body_size=6., trail_width=3., tInc_fac = 0.35, Max_PathLength=None, bint ShootingStars=True, bint Periodic=True, rel_trail_length_half_life = 0.03):
        """
        Plots 2D trajectories with one color per body and saves image in file
        """

        cdef Py_ssize_t ib, iint 
        cdef Py_ssize_t il, loop_id
        cdef double mass, line_width, point_size, mul

        cdef np.ndarray[double, ndim=1, mode='c'] alpha_arr

        assert self.geodim == 2
        assert self.geodim == allpos.shape[2]
        assert self.nbody == allpos.shape[0]

        npts = allpos.shape[1]

        ncol = len(color_list)

        if xlim is None:

            xmin, ymin = allpos.min(axis=(0,1))
            xmax, ymax = allpos.max(axis=(0,1))

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

        dx = xsup - xinf
        dy = ysup - yinf
        distance_ref = np.sqrt(dx*dx + dy*dy)

        if Max_PathLength is None:
            d_allpos = allpos[:,1:,:] - allpos[:,:-1,:]
            Max_PathLength = np.max(np.sum(np.hypot(d_allpos[:,:,0], d_allpos[:,:,1]), axis=1), axis=0)

        distance_rel = Max_PathLength / distance_ref

        tInc = tInc_fac / (fps * distance_rel) 

        # Plot-related
        fig = plt.figure(facecolor=bgColor)
        fig.set_size_inches(fig_size)
        fig.set_dpi(dpi)
        ax = plt.gca()

        plt_lines = []
        plt_varcolor_lines = []
        plt_points = []
        iplt = 0
        for ib in range(self.nbody):
            iplt += 1

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

            line_width = trail_width
            point_size = body_size * 25

            if Mass_Scale:
                mass = self._loopmass[self._bodyloop[ib]]
                mul = csqrt(mass)
                line_width = line_width * mul
                point_size = point_size * mul

            point_color = current_color
            if ShootingStars:
                lines_color = FallbackTrailColor
            else:
                lines_color = current_color

            plt_lines.append(ax.plot(allpos[ib,:,0], allpos[ib,:,1], color=lines_color, antialiased=True, zorder=-iplt, linewidth=line_width))

            if ShootingStars:

                transparent_color = matplotlib.colors.to_rgba(current_color, alpha=0.)

                segments = np.stack((allpos[ib,:-1,:], allpos[ib,1:,:]), axis=1)
                lc = matplotlib.collections.LineCollection(segments, colors=transparent_color, linewidths=line_width, antialiased=False)
                plt_varcolor_lines.append(ax.add_collection(lc))

            zorder = 3+iplt
            plt_points.append(ax.scatter(allpos[ib,0,0], allpos[ib,0,1], color=point_color, antialiased=True, zorder=zorder, marker = 'o', edgecolors='k', s=point_size))

        ax.axis('off')
        ax.set_xlim([xinf, xsup])
        ax.set_ylim([yinf, ysup ])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        if ShootingStars:
            alpha_arr = np.zeros((npts-1), dtype=np.float64)
        else:
            alpha_arr = None

        alpha_fac_in = 2**(-1./(npts*rel_trail_length_half_life))

        def update(i_frame, np.ndarray[double, ndim=1, mode='c'] alpha_buf = alpha_arr, double alpha_fac = alpha_fac_in):

            cdef double t = i_frame * tInc * allpos.shape[1]
            
            cdef Py_ssize_t tp = min(math.ceil(t), allpos.shape[1]-1)
            cdef Py_ssize_t tm = tp-1

            cdef double a = tp - t
            cdef double alpha = 1.

            cdef Py_ssize_t i, j
            cdef Py_ssize_t n = npts-1
            cdef Py_ssize_t no = n+tp-1

            if ShootingStars:

                if Periodic:
                    for i in range(n):
                        j = (no-i) % n
                        alpha_buf[j] = alpha
                        alpha *= alpha_fac
                else:
                    for i in range(tp):
                        j = tp-i
                        alpha_buf[j] = alpha
                        alpha *= alpha_fac

            for ib in range(self.nbody):
                pos = a * allpos[ib,tm,:] + (1.-a)*allpos[ib,tp,:]
                plt_points[ib].set_offsets(pos)

                if ShootingStars:
                    plt_varcolor_lines[ib].set_alpha(alpha_buf)
        
        n_frames = math.floor(1. / tInc)
        ani = matplotlib.animation.FuncAnimation(fig=fig, func=update, frames=n_frames)
        writer = matplotlib.animation.FFMpegWriter(fps=fps, codec="h264", extra_args=["-preset", "veryslow","-crf","0"])
        ani.save(filename, writer=writer)
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
                    if (2*self._params_shifts[il] < self.nparams_incl_o):
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
                    if (2*self._params_shifts[il] < self.nparams_incl_o):
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
                    if (2*self._params_shifts[il] < self.nparams_incl_o):
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
                    if (2*self._params_shifts[il] < self.nparams_incl_o):
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
            if (2*self._params_shifts[il] < self.nparams_incl_o):
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
            if (2*self._params_shifts[il] < self.nparams_incl_o):
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
    def segmpos_to_allsegm_noopt(self, double[:,:,::1] segmpos, bint pos=True):

        assert self.segm_store == segmpos.shape[1]

        cdef Py_ssize_t ib, iint, il
        cdef Py_ssize_t ibeg, segmend
        cdef Py_ssize_t segmbeg, iend
        cdef Py_ssize_t isegm_source, isegm_target
        cdef ActionSym Sym

        cdef np.ndarray[double, ndim=3, mode='c'] all_segmpos = np.empty((self.nsegm, self._nint, self.geodim), dtype=np.float64)

        for iint in range(self.nint_min):

            for isegm_target in range(self.nsegm):

                ib = self._intersegm_to_body[isegm_target]
                isegm_source = self._bodysegm[ib, iint]

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

                Sym.TransformSegment(segmpos[isegm_source,segmbeg:segmend,:], all_segmpos[isegm_target,ibeg:iend,:])

        return all_segmpos
        
    @cython.final
    def segmpos_to_allbody_noopt(self, double[:,:,::1] segmpos, bint pos=True):

        assert self.segm_store == segmpos.shape[1]

        cdef Py_ssize_t ib, iint
        cdef Py_ssize_t ibeg, segmend
        cdef Py_ssize_t segmbeg, iend
        cdef ActionSym Sym

        cdef np.ndarray[double, ndim=3, mode='c'] all_bodypos = np.empty((self.nbody, self._nint, self.geodim), dtype=np.float64)

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
            mass = self._segmmass[isegm]
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

        cdef Py_ssize_t il, isegm, ib, iint

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

        for il in range(self.nloop):
            ib = self._Targets[il,0]
            
            for iint in range(self.nint_min):
                isegm = self._bodysegm[ib, iint]
                out_loop_len_mv[il] += out_segm_len_mv[isegm]
            
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
    def ComputeCenterOfMass(self, double[:,:,::1] segmpos):

        # return np.einsum("qik,ik->q", np.asarray(self._CoMMat), np.asarray(segmpos).sum(axis=1))/self.segm_store

        cdef np.ndarray[double, ndim=2, mode='c'] segmpossum = np.asarray(segmpos).sum(axis=1)
        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty(self.geodim, dtype=np.float64) 
        
        cdef int geodim = self.geodim
        cdef int n = self.nsegm * self.geodim
        cdef double mul = (1./segmpos.shape[1])

        scipy.linalg.cython_blas.dgemv(transt,&n,&geodim,&mul,&self._CoMMat[0,0,0],&n,&segmpossum[0,0],&int_one,&zero_double,&res[0],&int_one)

        return res

    @cython.final
    def Compute_periodicity_default_pos(self, double[::1] xo, double[::1] xf):

        assert xo.shape[0] == self.nsegm * self.geodim
        assert xf.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.empty((self.nsegm * self.geodim), dtype=np.float64)

        cdef Py_ssize_t isegm, idim, jdim, i, j

        if self.TimeRev > 0:

            for isegm in range(self.nsegm):

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = xf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefEnd_SpaceRotPos[isegm,idim,jdim] * xo[j]
                        j += 1

                    i += 1

        else:

            for isegm in range(self.nsegm):

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

        if self.TimeRev > 0:

            for isegm in range(self.nsegm):

                i = isegm * self.geodim

                for idim in range(self.geodim):

                    res[i] = vf[i]

                    j = self._PerDefEnd_Isegm[isegm] * self.geodim
                    
                    for jdim in range(self.geodim):

                        res[i] -= self._PerDefEnd_SpaceRotVel[isegm,idim,jdim] * vo[j]
                        j += 1

                    i += 1

        else:

            for isegm in range(self.nsegm):

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

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.zeros((self.nsegm * self.geodim), dtype=np.float64)

        cdef Py_ssize_t isegm, idim, jdim, i, j

        if self.TimeRev < 0:

            for isegm in range(self.nsegm):

                    j = self._PerDefBeg_Isegm[isegm] * self.geodim

                    for idim in range(self.geodim):

                        res[j] += xo[j]
                        i = isegm * self.geodim

                        for jdim in range(self.geodim):

                            res[j] -= self._PerDefBeg_SpaceRotPos[isegm,idim,jdim] * xo[i]
                            i += 1

                        j += 1

        return res
        
    @cython.final
    def Compute_initial_constraint_default_vel(self, double[::1] vo):

        assert vo.shape[0] == self.nsegm * self.geodim

        cdef np.ndarray[double, ndim=1, mode='c'] res = np.zeros((self.nsegm * self.geodim), dtype=np.float64)

        cdef Py_ssize_t isegm, idim, jdim, i, j

        if self.TimeRev < 0:

            for isegm in range(self.nsegm):

                j = self._PerDefBeg_Isegm[isegm] * self.geodim

                for idim in range(self.geodim):

                    res[j] += vo[j]
                    i = isegm * self.geodim

                    for jdim in range(self.geodim):

                        res[j] -= self._PerDefBeg_SpaceRotVel[isegm,idim,jdim] * vo[i]
                        i += 1

                    j += 1

        return res

    @cython.final
    cpdef Compute_ODE_default(self, double[:,:,::1] xo, double[:,:,::1] vo, double[:,:,::1] xf, double[:,:,::1] vf):

        assert xo.shape[0] == self.segm_store
        assert xo.shape[1] == self.nsegm
        assert xo.shape[2] == self.geodim

        assert vo.shape[0] == self.segm_store
        assert vo.shape[1] == self.nsegm
        assert vo.shape[2] == self.geodim

        assert xf.shape[0] == self.segm_store
        assert xf.shape[1] == self.nsegm
        assert xf.shape[2] == self.geodim

        assert vf.shape[0] == self.segm_store
        assert vf.shape[1] == self.nsegm
        assert vf.shape[2] == self.geodim

        cdef int nelem

        cdef Py_ssize_t isegm, jsegm
        cdef Py_ssize_t idim, jdim
        cdef Py_ssize_t iint, iend

        iend = self.segm_store-1

        cdef np.ndarray[double, ndim=3, mode='c'] dx_np = np.empty((self.segm_store, self.nsegm, self.geodim), dtype=np.float64)
        cdef np.ndarray[double, ndim=3, mode='c'] dv_np = np.empty((self.segm_store, self.nsegm, self.geodim), dtype=np.float64)

        cdef double[:,:,::1] dx = dx_np
        cdef double[:,:,::1] dv = dv_np

        if self.TimeRev > 0:
            iint = 0
        else:
            iint = iend

        for isegm in range(self.nsegm):

            jsegm = self._PerDefEnd_Isegm[isegm]

            for idim in range(self.geodim):

                dx[0,isegm,idim] = xf[iend,isegm,idim]
                for jdim in range(self.geodim):
                    dx[0,isegm,idim] -= self._PerDefEnd_SpaceRotPos[isegm,idim,jdim] * xo[iint,jsegm,jdim]

                dv[0,isegm,idim] = vf[iend,isegm,idim]
                for jdim in range(self.geodim):
                    dv[0,isegm,idim] -= self._PerDefEnd_SpaceRotVel[isegm,idim,jdim] * vo[iint,jsegm,jdim]

        nelem = (self.segm_store-1) * self.nsegm * self.geodim

        scipy.linalg.cython_blas.dcopy(&nelem,&xf[1,0,0],&int_one,&dx[1,0,0],&int_one)
        scipy.linalg.cython_blas.dcopy(&nelem,&vf[1,0,0],&int_one,&dv[1,0,0],&int_one)

        scipy.linalg.cython_blas.daxpy(&nelem,&minusone_double,&xo[1,0,0],&int_one,&dx[1,0,0],&int_one)
        scipy.linalg.cython_blas.daxpy(&nelem,&minusone_double,&vo[1,0,0],&int_one,&dv[1,0,0],&int_one)

        return dx_np, dv_np

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
            &pos_flat[0]                            , &res[0]                               ,
            self.nbin_segm_unique                   , self.geodim                           ,   
            self.nsegm                              , 1                                     ,           
            &self._BinSourceSegm[0]                 , &self._BinTargetSegm[0]               ,
            &self._BinSpaceRot[0,0,0]               , &self._BinSpaceRotIsId[0]             ,
            &self._BinProdChargeSumSource_ODE[0]    , &self._BinProdChargeSumTarget_ODE[0]  ,
            self._inter_law                         , self._inter_law_param_ptr             ,
        )

        return res

    @cython.final
    def Compute_grad_forces(self, double t, double[::1] pos_flat, double[:,::1] dpos_flat):

        assert pos_flat.shape[0] == self.nsegm * self.geodim

        cdef Py_ssize_t grad_ndof = dpos_flat.shape[1]
        cdef np.ndarray[double, ndim=2, mode='c'] res = np.empty((self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_forces_vectorized(
            &pos_flat[0]                            , &dpos_flat[0,0]                       , &res[0,0] ,
            self.nbin_segm_unique                   , self.geodim                           ,   
            self.nsegm                              , 1                                     , grad_ndof ,
            &self._BinSourceSegm[0]                 , &self._BinTargetSegm[0]               ,
            &self._BinSpaceRot[0,0,0]               , &self._BinSpaceRotIsId[0]             ,
            &self._BinProdChargeSumSource_ODE[0]    , &self._BinProdChargeSumTarget_ODE[0]  ,
            self._inter_law                         , self._inter_law_param_ptr             ,
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
            &pos_flat[0,0]                          , &res[0,0]                             ,
            self.nbin_segm_unique                   , self.geodim                           , 
            self.nsegm                              , nvec                                  ,       
            &self._BinSourceSegm[0]                 , &self._BinTargetSegm[0]               ,
            &self._BinSpaceRot[0,0,0]               , &self._BinSpaceRotIsId[0]             ,
            &self._BinProdChargeSumSource_ODE[0]    , &self._BinProdChargeSumTarget_ODE[0]  ,
            self._inter_law                         , self._inter_law_param_ptr             ,
        )

        return res

    @cython.final
    def Compute_grad_forces_vectorized(self, double[::1] t, double[:,::1] pos_flat, double[:,:,::1] dpos_flat):

        assert pos_flat.shape[1] == self.nsegm * self.geodim

        cdef Py_ssize_t nvec = pos_flat.shape[0]
        cdef Py_ssize_t grad_ndof = dpos_flat.shape[2]
        cdef np.ndarray[double, ndim=3, mode='c'] res = np.empty((nvec, self.nsegm * self.geodim, grad_ndof), dtype=np.float64)

        Compute_grad_forces_vectorized(
            &pos_flat[0,0]                          , &dpos_flat[0,0,0]                     , &res[0,0,0]   ,
            self.nbin_segm_unique                   , self.geodim                           , 
            self.nsegm                              , nvec                                  , grad_ndof     ,
            &self._BinSourceSegm[0]                 , &self._BinTargetSegm[0]               ,
            &self._BinSpaceRot[0,0,0]               , &self._BinSpaceRotIsId[0]             ,
            &self._BinProdChargeSumSource_ODE[0]    , &self._BinProdChargeSumTarget_ODE[0]  ,
            self._inter_law                         , self._inter_law_param_ptr             ,
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
    def Get_ODE_def(self, double[::1] params_mom_buf = None, vector_calls = True, LowLevel = True, NoSymIfPossible = True, grad = False, regular_init = False):

        dict_res = {
            "t_span" : (0., 1./self.nint_min)   ,
            "vector_calls" : vector_calls       ,
        }

        if params_mom_buf is not None:
            if regular_init:
                segmpos = self.params_to_segmpos(params_mom_buf)
                segmmom = self.params_to_segmmom(params_mom_buf)
                dict_res["reg_xo"] = np.ascontiguousarray(segmpos.swapaxes(0, 1).reshape(self.segm_store,-1))
                dict_res["reg_vo"] = np.ascontiguousarray(segmmom.swapaxes(0, 1).reshape(self.segm_store,-1))
            
            else:
                dict_res["xo"], dict_res["vo"] = self.Compute_init_pos_mom(params_mom_buf) 

        NoSymPossible = self.BinSpaceRotIsId.all()
        NoSym = NoSymIfPossible and NoSymPossible

        if LowLevel:

            user_data = self.Get_ODE_params()

            if vector_calls:

                dict_res["fun"] = scipy.LowLevelCallable.from_cython(
                    choreo.cython._NBodySyst_ann                ,
                    "Compute_velocities_vectorized_user_data"   ,
                    user_data                                   ,
                )
                if grad:
                    dict_res["grad_fun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst_ann                    ,
                        "Compute_grad_velocities_vectorized_user_data"  ,
                        user_data                                       ,
                    )

                if NoSym:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst_ann                ,
                        "Compute_forces_vectorized_nosym_user_data" ,
                        user_data                                   ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst_ann                        ,
                            "Compute_grad_forces_vectorized_nosym_user_data"    ,
                            user_data                                           ,
                        )

                else:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst_ann                ,
                        "Compute_forces_vectorized_user_data"       ,
                        user_data                                   ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst_ann                ,
                            "Compute_grad_forces_vectorized_user_data"  ,
                            user_data                                   ,
                        )

            else:

                dict_res["fun"] = scipy.LowLevelCallable.from_cython(
                    choreo.cython._NBodySyst_ann                ,
                    "Compute_velocities_user_data"              ,
                    user_data                                   ,
                )
                if grad:
                    dict_res["grad_fun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst_ann                ,
                        "Compute_grad_velocities_user_data"         ,
                        user_data                                   ,
                    )

                if NoSym:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst_ann            ,
                        "Compute_forces_nosym_user_data"        ,
                        user_data                               ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst_ann            ,
                            "Compute_grad_forces_nosym_user_data"   ,
                            user_data                               ,
                        )

                else:
                    dict_res["gun"] = scipy.LowLevelCallable.from_cython(
                        choreo.cython._NBodySyst_ann            ,
                        "Compute_forces_user_data"              ,
                        user_data                               ,
                    )
                    if grad:
                        dict_res["grad_gun"] = scipy.LowLevelCallable.from_cython(
                            choreo.cython._NBodySyst_ann    ,
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

    @cython.final
    @cython.cdivision(True)
    def PropagateMonodromy_noopt(self, double[:,:,::1] segmpos_grad_ODE, double[:,:,::1] segmmom_grad_ODE, bint OnlyFinal = True):
        # Expects segmpos_grad_ODE and segmmom_grad_ODE generated with keep_init = False, and t_span = (0., 1./nint_min). Only final values.

        cdef Py_ssize_t iint
        cdef Py_ssize_t i, j, k
        cdef Py_ssize_t ib, jb, kb
        cdef Py_ssize_t isegm_source, isegm_target
        cdef Py_ssize_t jsegm_source, jsegm_target
        cdef Py_ssize_t ksegm_source, ksegm_target
        cdef Py_ssize_t n = self.nsegm * self.geodim

        cdef Py_ssize_t idim, jdim, kdim

        cdef ActionSym Sym_target_i, Sym_target_k

        cdef int nelem

        assert segmpos_grad_ODE.shape[0] == 1
        assert segmpos_grad_ODE.shape[1] == n
        assert segmpos_grad_ODE.shape[2] == 2*n

        assert segmmom_grad_ODE.shape[0] == 1
        assert segmmom_grad_ODE.shape[1] == n
        assert segmmom_grad_ODE.shape[2] == 2*n

        cdef np.ndarray[double, ndim=7, mode='c'] MonodromyMat_np = np.zeros((self.nint_min, 2, self.nsegm, self.geodim, 2, self.nsegm, self.geodim), dtype=np.float64)        
        cdef double[:,:,:,:,:,:,::1] MonodromyMat = MonodromyMat_np
        
        nelem = 2*n*n
        scipy.linalg.cython_blas.dcopy(&nelem,&segmpos_grad_ODE[0,0,0],&int_one,&MonodromyMat[0,0,0,0,0,0,0],&int_one)
        scipy.linalg.cython_blas.dcopy(&nelem,&segmmom_grad_ODE[0,0,0],&int_one,&MonodromyMat[0,1,0,0,0,0,0],&int_one)    

        cdef double[:,:,:,:,:,::1] MonodromyMat_in = MonodromyMat_np[0,:,:,:,:,:,:]

        for iint in range(1, self.nint_min):

            for isegm_target in range(self.nsegm):

                ib = self._intersegm_to_body[isegm_target]
                isegm_source = self._bodysegm[ib, iint]

                Sym_target_i = self.intersegm_to_all[ib][iint]
                R_target_i = Sym_target_i.SpaceRot

                for ksegm_target in range(self.nsegm):

                    kb = self._intersegm_to_body[ksegm_target]
                    ksegm_source = self._bodysegm[kb, iint]

                    Sym_target_k = self.intersegm_to_all[kb][iint]
                    R_target_k = Sym_target_k.SpaceRot

                    for jsegm_target in range(self.nsegm):

                        jb = self._intersegm_to_body[jsegm_target]
                        jsegm_source = self._bodysegm[jb, iint]

                        for i in range(2):
                            for j in range(2):
                                for k in range(2):

                                    if ((iint % 2) == 0) or (self.TimeRev > 0):
                                        MM_source = np.asarray(MonodromyMat_in[i,isegm_source,:,k,ksegm_source,:])

                                    else:
                                        # Circumvents the computation of the inverse of MonodromyMat_in, taking advantage of symplecticity
                                        if (i==0) == (k==0):
                                            MM_source = np.asarray(MonodromyMat_in[1-i,ksegm_source,:,1-k,isegm_source,:]).T
                                        else:
                                            MM_source = np.asarray(MonodromyMat_in[i,ksegm_source,:,k,isegm_source,:]).T

                                    MM_target = np.asarray(MonodromyMat[iint-1,k,ksegm_target ,:,j,jsegm_target,:])

                                    MM = R_target_i @ MM_source @ R_target_k.T @ MM_target 

                                    for idim in range(self.geodim):
                                        for jdim in range(self.geodim):

                                            MonodromyMat[iint,i,isegm_target,idim,j,jsegm_target,jdim] += MM[idim,jdim]

        if OnlyFinal:
            return np.asarray(MonodromyMat_np[self.nint_min-1,:,:,:,:,:,:])
        else:
            return MonodromyMat_np

    @cython.final
    @cython.cdivision(True)
    def PropagateMonodromy(self, double[:,:,::1] segmpos_grad_ODE, double[:,:,::1] segmmom_grad_ODE):
        # Expects segmpos_grad_ODE and segmmom_grad_ODE generated with keep_init = False, and t_span = (0., 1./nint_min). Only final value

        cdef Py_ssize_t iint
        cdef Py_ssize_t i, j, k
        cdef Py_ssize_t ib, jb, kb
        cdef Py_ssize_t isegm_source, isegm_target
        cdef Py_ssize_t jsegm_source, jsegm_target
        cdef Py_ssize_t ksegm_source, ksegm_target
        cdef int n = self.nsegm * self.geodim

        cdef ActionSym Sym_target_i, Sym_target_k

        cdef int nelem

        assert segmpos_grad_ODE.shape[0] == 1
        assert segmpos_grad_ODE.shape[1] == n
        assert segmpos_grad_ODE.shape[2] == 2*n

        assert segmmom_grad_ODE.shape[0] == 1
        assert segmmom_grad_ODE.shape[1] == n
        assert segmmom_grad_ODE.shape[2] == 2*n
      
        cdef double[:,:,:,:,:,::1] MonodromyMat_prev = np.empty((2, self.nsegm, self.geodim, 2, self.nsegm, self.geodim), dtype=np.float64)        
        cdef double[:,:,:,:,:,::1] MonodromyMat = np.empty((2, self.nsegm, self.geodim, 2, self.nsegm, self.geodim), dtype=np.float64)        
        nelem = 2*n*n
        scipy.linalg.cython_blas.dcopy(&nelem,&segmpos_grad_ODE[0,0,0],&int_one,&MonodromyMat[0,0,0,0,0,0],&int_one)
        scipy.linalg.cython_blas.dcopy(&nelem,&segmmom_grad_ODE[0,0,0],&int_one,&MonodromyMat[1,0,0,0,0,0],&int_one)    

        cdef double[:,:,:,:,::1] segmpos_grad_in = <double[:self.nsegm,:self.geodim,:2,:self.nsegm,:self.geodim:1]> &segmpos_grad_ODE[0,0,0]
        cdef double[:,:,:,:,::1] segmmom_grad_in = <double[:self.nsegm,:self.geodim,:2,:self.nsegm,:self.geodim:1]> &segmmom_grad_ODE[0,0,0]

        cdef double* R_target_i_ptr
        cdef double* MM_source_ptr
        cdef double* R_target_k_ptr
        cdef double* MM_target_ptr

        cdef int geodim = self.geodim
        cdef int ld = 2*n
        cdef char* trans_MM 

        cdef double* buf_1  = <double*> malloc(sizeof(double)*geodim*geodim)
        cdef double* buf_2  = <double*> malloc(sizeof(double)*geodim*geodim)

        for iint in range(1, self.nint_min):

            MonodromyMat, MonodromyMat_prev = MonodromyMat_prev, MonodromyMat
            memset(&MonodromyMat[0,0,0,0,0,0], 0, sizeof(double)*4*n*n)

            for isegm_target in range(self.nsegm):

                ib = self._intersegm_to_body[isegm_target]
                isegm_source = self._bodysegm[ib, iint]

                Sym_target_i = self.intersegm_to_all[ib][iint]
                R_target_i_ptr = &Sym_target_i._SpaceRot[0,0]

                for ksegm_target in range(self.nsegm):

                    kb = self._intersegm_to_body[ksegm_target]
                    ksegm_source = self._bodysegm[kb, iint]

                    Sym_target_k = self.intersegm_to_all[kb][iint]
                    R_target_k_ptr = &Sym_target_k._SpaceRot[0,0]   

                    for jsegm_target in range(self.nsegm):

                        jb = self._intersegm_to_body[jsegm_target]
                        jsegm_source = self._bodysegm[jb, iint]
                        
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):

                                    if ((iint % 2) == 0) or (self.TimeRev > 0):

                                        if (i == 0):
                                            MM_source_ptr = &segmpos_grad_in[isegm_source,0,k,ksegm_source,0]
                                        else:
                                            MM_source_ptr = &segmmom_grad_in[isegm_source,0,k,ksegm_source,0]

                                        trans_MM = transn

                                    else:
                                        # Circumvents the computation of the inverse of MonodromyMat_in, taking advantage of symplecticity
                                        if (k == 0):
                                            MM_source_ptr = &segmmom_grad_in[ksegm_source,0,1-i,isegm_source,0]
                                        else:
                                            MM_source_ptr = &segmpos_grad_in[ksegm_source,0,1-i,isegm_source,0]

                                        trans_MM = transt

                                    MM_target_ptr = &MonodromyMat_prev[k,ksegm_target,0,j,jsegm_target,0]

                                    # buf_1 = R_target_k.T . MM_target 
                                    scipy.linalg.cython_blas.dgemm(transn,transt,&geodim,&geodim,&geodim,&one_double,MM_target_ptr,&ld,R_target_k_ptr,&geodim,&zero_double,buf_1,&geodim)

                                    # buf_2 = MM_source . buf_1
                                    scipy.linalg.cython_blas.dgemm(transn,trans_MM,&geodim,&geodim,&geodim,&one_double,buf_1,&geodim,MM_source_ptr,&ld,&zero_double,buf_2,&geodim)

                                    # MonodromyMat += R_target_i . buf_2
                                    scipy.linalg.cython_blas.dgemm(transn,transn,&geodim,&geodim,&geodim,&one_double,buf_2,&geodim,R_target_i_ptr,&geodim,&one_double,&MonodromyMat[i,isegm_target,0,j,jsegm_target,0],&ld)

        free(buf_1)
        free(buf_2)

        return np.asarray(MonodromyMat)
