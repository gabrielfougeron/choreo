'''
quad.pyx : Defines segment quadrature related things.

'''

__all__ = [
    'QuadTable'             ,
    'IntegrateOnSegment'    ,
]

from choreo.segm.cython.eft_lib cimport TwoSum_incr

cimport scipy.linalg.cython_blas
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

cimport cython

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf

from choreo.scipy_plus.cython.ccallback cimport ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS, ccallback_signature_t

from choreo.scipy_plus.cython.blas_consts cimport *

cdef int PY_FUN = -1
cdef int C_FUN_MEMORYVIEW = 0
cdef int C_FUN_MEMORYVIEW_DATA = 1
cdef int C_FUN_POINTER = 2
cdef int C_FUN_POINTER_DATA = 3
cdef int C_FUN_ONEVAL = 4
cdef int C_FUN_ONEVAL_DATA = 5
cdef int N_SIGNATURES = 6
cdef ccallback_signature_t signatures[7]

ctypedef void (*c_fun_type_memoryview)(double, double[::1]) noexcept nogil 
signatures[C_FUN_MEMORYVIEW].signature = b"void (double, __Pyx_memviewslice)"
signatures[C_FUN_MEMORYVIEW].value = C_FUN_MEMORYVIEW

ctypedef void (*c_fun_type_memoryview_data)(double, double[::1], void*) noexcept nogil 
signatures[C_FUN_MEMORYVIEW_DATA].signature = b"void (double, __Pyx_memviewslice, void *)"
signatures[C_FUN_MEMORYVIEW_DATA].value = C_FUN_MEMORYVIEW_DATA

ctypedef void (*c_fun_type_pointer)(double, double*) noexcept nogil 
signatures[C_FUN_POINTER].signature = b"void (double, double *)"
signatures[C_FUN_POINTER].value = C_FUN_POINTER

ctypedef void (*c_fun_type_pointer_data)(double, double*, void*) noexcept nogil 
signatures[C_FUN_POINTER_DATA].signature = b"void (double, double *, void *)"
signatures[C_FUN_POINTER_DATA].value = C_FUN_POINTER_DATA

ctypedef double (*c_fun_type_oneval)(double) noexcept nogil 
signatures[C_FUN_ONEVAL].signature = b"double (double)"
signatures[C_FUN_ONEVAL].value = C_FUN_ONEVAL

ctypedef double (*c_fun_type_oneval_data)(double, void*) noexcept nogil 
signatures[C_FUN_ONEVAL_DATA].signature = b"double (double, void *)"
signatures[C_FUN_ONEVAL_DATA].value = C_FUN_ONEVAL_DATA

signatures[N_SIGNATURES].signature = NULL

cdef inline void LowLevelFun_apply(
    ccallback_t callback    ,
    double x                ,
    double[::1] res         ,
) noexcept nogil:

    if (callback.py_function == NULL):

        if (callback.user_data == NULL):

            if callback.signature.value == C_FUN_MEMORYVIEW:
                (<c_fun_type_memoryview> callback.c_function)(x, res)

            elif callback.signature.value == C_FUN_POINTER:
                (<c_fun_type_pointer> callback.c_function)(x, &res[0])

            elif callback.signature.value == C_FUN_ONEVAL:
                res[0] = (<c_fun_type_oneval> callback.c_function)(x)

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")

        else:

            if callback.signature.value == C_FUN_MEMORYVIEW_DATA:
                (<c_fun_type_memoryview_data> callback.c_function)(x, res, callback.user_data)

            elif callback.signature.value == C_FUN_POINTER_DATA:
                (<c_fun_type_pointer_data> callback.c_function)(x, &res[0], callback.user_data)

            elif callback.signature.value == C_FUN_ONEVAL_DATA:
                res[0] = (<c_fun_type_oneval_data> callback.c_function)(x, callback.user_data)

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")

    else:

        with gil:

            PyFun_apply(callback, x, res)

cdef void PyFun_apply(
    ccallback_t callback    ,
    double x                ,
    double[::1] res         ,
):

    cdef int n = res.shape[0]
    cdef double[::1] res_1D = (<object> callback.py_function)(x)

    scipy.linalg.cython_blas.dcopy(&n,&res_1D[0],&int_one,&res[0],&int_one)

@cython.final
cdef class QuadTable:
    r"""Numerical integration and approximation.

    This class implements useful methods for the approximate integration (or quadrature) of regular functions on a segment, as well as other related numerical methods.

    .. math::
        \int_{0}^{1} f(x)\  \mathrm{d}x \approx \sum_{i=0}^{n-1} w_i f(x_i)
    
    """

    def __init__(
        self                ,
        w                   ,
        x                   ,
        wlag                ,
        th_cvg_rate = None  ,
    ):
        """Builds a :class:`QuadTable` from input node and weight arrays.

        Parameters
        ----------
        w : :class:`numpy:numpy.ndarray`:class:`(shape = (n), dtype = np.float64)`
            Quadrature weights.
        x : :class:`numpy:numpy.ndarray`:class:`(shape = (n), dtype = np.float64)`
            Quadrature nodes on :math:`[0,1]`.
        wlag : :class:`numpy:numpy.ndarray`:class:`(shape = (n), dtype = np.float64)`
            Barycentric Lagrange interpolation weights.
        th_cvg_rate : :class:`python:int`, optional
            Theoretical convergence rate of the quadrature formula, by default :data:`python:None`.
        """    

        self._w = w.copy()
        self._x = x.copy()
        self._wlag = wlag.copy()

        assert self._w.shape[0] == self._x.shape[0]
        assert self._wlag.shape[0] == self._x.shape[0]

        if th_cvg_rate is None:
            self._th_cvg_rate = -1
        else:
            self._th_cvg_rate = th_cvg_rate

    def __repr__(self):

        res = f'QuadTable object with {self._w.shape[0]} nodes\n'
        res += f'Nodes: {self.x}\n'
        res += f'Weights: {self.w}\n'
        res += f'Barycentric Lagrange interpolation weights: {self.wlag}\n'

        return res

    @property
    def nsteps(self):
        """Number of steps (*i.e.* functions evaluations required) of the method."""
        return self._w.shape[0]

    @property
    def x(self):
        """Nodes of the method in :math:`[-1,1]`."""
        return np.asarray(self._x)
    
    @property
    def w(self):
        """Integration weights of the method on :math:`[-1,1]`."""
        return np.asarray(self._w)      

    @property
    def wlag(self):
        """Barycentric Lagrange interpolation weights relative to the nodes."""
        return np.asarray(self._wlag)    

    @property
    def th_cvg_rate(self):
        """Theoretical congergence rate of the quadrature for smooth functions."""
        return self._th_cvg_rate

    @cython.final
    cpdef QuadTable symmetric_adjoint(self):
        r"""Computes the symmetric adjoint of a :class:`QuadTable`.

        The symmetric adjoint applied to :math:`x\mapsto f(1-x)` gives the same result as the original method applied to :math:`f`.

        Returns
        -------
        :class:`choreo.segm.quad.QuadTable`
            The adjoint quadrature.
        """ 

        cdef Py_ssize_t n = self._w.shape[0]
        cdef Py_ssize_t i, j

        cdef double[::1] w_sym = np.empty((n), dtype=np.float64)
        cdef double[::1] x_sym = np.empty((n), dtype=np.float64)
        cdef double[::1] wlag_sym = np.empty((n), dtype=np.float64)

        for i in range(n):

            w_sym[i] = self._w[n-1-i]
            x_sym[i] = 1. - self._x[n-1-i]
            wlag_sym[i] = self._wlag[n-1-i]

        return QuadTable(
            w_sym               ,
            x_sym               ,
            wlag_sym            ,
            self._th_cvg_rate   ,
        )

    @cython.final
    cdef double _symmetry_default(
        self            ,
        QuadTable other ,
    ) noexcept nogil:

        cdef Py_ssize_t nsteps = self._w.shape[0]
        cdef Py_ssize_t i,j
        cdef double maxi = -1
        cdef double val

        for i in range(nsteps):

            val = self._w[i] - other._w[nsteps-1-i] 
            maxi = max(maxi, cfabs(val))

            val = self._x[i] + other._x[nsteps-1-i] - 1
            maxi = max(maxi, cfabs(val))

        return maxi    

    @cython.final
    def symmetry_default(
        self        ,
        other = None,
    ):
        """Computes the symmetry default of a single / a pair of :class:`QuadTable`.

        A method is said to be symmetric if its symmetry default is zero, namely if it coincides with its :meth:`symmetric_adjoint`.

        Example
        -------

        >>> import choreo
        >>> Gauss = choreo.segm.multiprec_tables.ComputeQuadrature(10, method="Gauss")
        >>> Gauss.symmetry_default()
        0.0
        >>> Radau_I = choreo.segm.multiprec_tables.ComputeQuadrature(10, method="Radau_I")
        >>> Radau_I.symmetry_default()
        0.08008763577630496
        >>> Radau_I.symmetry_default(Radau_I.symmetric_adjoint())
        0.0

        Parameters
        ----------
        other : :class:`QuadTable`, optional
            By default :data:`python:None`.

        Returns
        -------
        :obj:`numpy:numpy.float64`
            The maximum symmetry violation.
        """    

        if other is None:
            return self._symmetry_default(self)
        else:
            return self._symmetry_default(other)
    
    @cython.final
    cdef bint _is_symmetric_pair(self, QuadTable other, double tol) noexcept nogil:
        return (self._symmetry_default(other) < tol)

    @cython.final
    def is_symmetric_pair(self, QuadTable other, double tol = 1e-12):
        return self._is_symmetric_pair(other, tol)

    @cython.final
    def is_symmetric(self, double tol = 1e-12):
        return self._is_symmetric_pair(self, tol)        

cpdef np.ndarray[double, ndim=1, mode="c"] IntegrateOnSegment(
    object fun              ,
    int ndim                ,
    (double, double) x_span ,
    QuadTable quad          ,
    Py_ssize_t nint = 1     ,
    bint DoEFT = True       ,
):
    r""" Computes an approximation of the integral of a function on a segment.

    The integrand can either be:

    * A `Python <https://www.python.org/>`_ function taking a :obj:`numpy:numpy.float64`, and returning a :class:`numpy.ndarray`:class:`(shape=ndim, dtype=np.float64)`.
    * A :class:`scipy:scipy.LowLevelCallable` for performance-critical use cases. See :ref:`sphx_glr__build_auto_examples_benchmarks_quad_integration_lowlevel_bench.py` for an in-depth example and comparison of the different alternatives.

    Example
    -------

    Let us compare the approximation and exact value of the `Wallis integral <https://en.wikipedia.org/wiki/Wallis%27_integrals>`_ of order 7.

    .. math::
        \int_0^{\frac{\pi}{2}} \operatorname{sin}^7(x)\ \mathrm{d}x = \frac{16}{35}

    >>> import numpy as np
    >>> import choreo
    >>> Gauss = choreo.segm.multiprec_tables.ComputeQuadrature(10, method="Gauss")
    >>> def fun(x):
    ...     return np.array([np.sin(x)**7])
    ...
    >>> np.abs(choreo.segm.quad.IntegrateOnSegment(fun, ndim=1, x_span=(0.,np.pi/2), quad=Gauss) - 16/35)
    array([4.65549821e-12])    

    Parameters
    ----------
    fun : :class:`python:object` or :class:`scipy:scipy.LowLevelCallable`
        Function to be integrated.
    ndim : :class:`python:int`
        Number of output dimensions of the integrand.
    x_span : (:obj:`numpy:numpy.float64`, :obj:`numpy:numpy.float64`)
        Lower and upper bound of the integration interval.
    quad : :class:`QuadTable`
        Weights and nodes of the integration.
    nint : :class:`python:int`, optional
        Number of sub-intervals for the integration, by default ``1``.
    DoEFT : :class:`python:bool`, optional
        Whether to , by default :data:`python:True`.
    """





    cdef ccallback_t callback
    ccallback_prepare(&callback, signatures, fun, CCALLBACK_DEFAULTS)

    cdef double[::1] f_res = np.empty((ndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] f_int_np = np.zeros((ndim),dtype=np.float64)
    cdef double[::1] f_int = f_int_np

    with nogil:
        IntegrateOnSegment_ann(
            callback    ,
            ndim        ,
            x_span      ,
            nint        ,
            DoEFT       ,
            quad._w     ,
            quad._x     ,
            f_res       ,
            f_int       ,
        )

    ccallback_release(&callback)

    return f_int_np

@cython.cdivision(True)
cdef void IntegrateOnSegment_ann(
    ccallback_t callback    ,
    int ndim                ,
    (double, double) x_span ,
    Py_ssize_t nint         ,
    bint DoEFT              ,
    double[::1] w           ,
    double[::1] x           ,
    double[::1] f_res       ,
    double[::1] f_int       ,
) noexcept nogil:

    cdef Py_ssize_t istep
    cdef Py_ssize_t iint
    cdef Py_ssize_t idim
    cdef double xbeg, dx
    cdef double xi

    cdef double* f_eft_comp

    if DoEFT:

        f_eft_comp = <double *> malloc(sizeof(double) * ndim)
        for istep in range(ndim):
            f_eft_comp[istep] = 0.

    cdef double *cdx = <double*> malloc(sizeof(double)*w.shape[0])

    dx = (x_span[1] - x_span[0]) / nint

    for istep in range(w.shape[0]):
        cdx[istep] = x[istep] * dx

    for iint in range(nint):
        xbeg = x_span[0] + iint * dx

        for istep in range(w.shape[0]):

            xi = xbeg + cdx[istep]

            # f_res = f(xi)
            LowLevelFun_apply(callback, xi, f_res)

            # f_int = f_int + w * f_res
            if DoEFT:
                scipy.linalg.cython_blas.dscal(&ndim,&w[istep],&f_res[0],&int_one)
                TwoSum_incr(&f_int[0],&f_res[0],f_eft_comp,ndim)

            else:
                scipy.linalg.cython_blas.daxpy(&ndim,&w[istep],&f_res[0],&int_one,&f_int[0],&int_one)

    scipy.linalg.cython_blas.dscal(&ndim,&dx,&f_int[0],&int_one)

    free(cdx)

    if DoEFT:
        free(f_eft_comp)
