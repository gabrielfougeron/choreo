'''
quad.pyx : Defines segment quadrature related things.

'''

__all__ = [
    'QuadTable'             ,
    'EvalOnNodes'           ,
    'IntegrateOnSegment'    ,
    'InterpolateOnSegment'  ,
]

from choreo.segm.cython.eft_lib cimport TwoSum_incr

cimport scipy.linalg.cython_blas
from libc.math cimport fabs as cfabs
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "float.h":
    double DBL_MAX

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
        \int_{0}^{1} f(x) \dd x \approx \sum_{i=0}^{n-1} w_i f(x_i)
        :label: QuadTable_int_approx

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
        """Number of steps of the method.
        
        This is the number of function evaluations needed to approximate an integral.
        """
        return self._w.shape[0]

    @property
    def x(self):
        """Nodes of the method in :math:`[0,1]`."""
        return np.asarray(self._x)
    
    @property
    def w(self):
        """Integration weights of the method on :math:`[0,1]`."""
        return np.asarray(self._w)      

    @property
    def wlag(self):
        """Barycentric Lagrange interpolation weights relative to the nodes."""
        return np.asarray(self._wlag)    

    @property
    def th_cvg_rate(self):
        """Theoretical convergence rate of the quadrature for smooth functions."""
        return self._th_cvg_rate

    @cython.final
    cpdef QuadTable symmetric_adjoint(self):
        r"""Computes the symmetric adjoint of a :class:`QuadTable`.

        The symmetric adjoint applied to :math:`x\mapsto f(1-x)` gives the same result as the original method applied to :math:`f`.

        Example
        -------
        >>> import choreo
        >>> Radau_I.symmetry_default(Radau_I.symmetric_adjoint())
        True

        See Also
        --------

        * :meth:`symmetry_default`
        * :meth:`is_symmetric_pair`

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
            wlag_sym[i] = - self._wlag[n-1-i]

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

            val = self._wlag[i] + other._wlag[nsteps-1-i]
            maxi = max(maxi, cfabs(val))

            val = self._x[i] + other._x[nsteps-1-i] - 1
            maxi = max(maxi, cfabs(val))

        return maxi    

    @cython.final
    def symmetry_default(
        self                    ,
        QuadTable other = None  ,
    ):
        """Computes the symmetry default of a single / a pair of :class:`QuadTable`.

        A method is said to be symmetric if its symmetry default is zero, namely if it coincides with its :meth:`symmetric_adjoint`.
        If the two methods do not have the same number of steps, the symmetry default is infinite by convention.

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

        See Also
        --------

        * :meth:`is_symmetric`
        * :meth:`is_symmetric_pair`

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
            if self._w.shape[0] == other._w.shape[0]:
                return self._symmetry_default(other)
            else:
                return np.inf
    
    @cython.final
    cdef bint _is_symmetric_pair(self, QuadTable other, double tol) noexcept nogil:
        return (self._symmetry_default(other) < tol)

    @cython.final
    def is_symmetric_pair(self, QuadTable other, double tol = 1e-12):
        r"""Returns :data:`python:True` if the pair of integration methods is symmetric.

        The pair of methods ``(self, other)`` is inferred symmetric if its symmetry default falls under the specified tolerance ``tol``.

        Example
        -------

        >>> import choreo
        >>> Radau_I = choreo.segm.multiprec_tables.ComputeQuadrature(10, method="Radau_I")
        >>> Radau_II = choreo.segm.multiprec_tables.ComputeQuadrature(10, method="Radau_II")
        >>> Radau_I.is_symmetric_pair(Radau_I)
        False
        >>> Radau_I.is_symmetric_pair(Radau_II)
        True

        See Also
        --------

        * :meth:`symmetry_default`
        * :meth:`is_symmetric`

        Parameters
        ----------
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symmetry default, by default ``1e-12``.        

        Returns
        -------
        :class:`python:bool`
            Whether the method is symmetric given the tolerance ``tol``.
        """ 

        if self._w.shape[0] == other._w.shape[0]:
            return self._is_symmetric_pair(other, tol)
        else:
            return False
        

    @cython.final
    def is_symmetric(self, double tol = 1e-12):
        r"""Returns :data:`python:True` if the integration method is symmetric.

        The method is inferred symmetric if its symmetry default falls under the specified tolerance ``tol``.

        Example
        -------

        >>> import choreo
        >>> Gauss = choreo.segm.multiprec_tables.ComputeQuadrature(10, method="Gauss")
        >>> Gauss.is_symmetric()
        True
        >>> Radau_I = choreo.segm.multiprec_tables.ComputeQuadrature(10, method="Radau_I")
        >>> Radau_I.is_symmetric()
        False

        See Also
        --------

        * :meth:`symmetry_default`
        * :meth:`is_symmetric_pair`

        Parameters
        ----------
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symmetry default, by default ``1e-12``.        

        Returns
        -------
        :class:`python:bool`
            Whether the method is symmetric given the tolerance ``tol``.
        """ 

        return self._is_symmetric_pair(self, tol)        

cpdef np.ndarray[double, ndim=2, mode="c"] EvalOnNodes(
    object fun              ,
    Py_ssize_t ndim         ,
    (double, double) x_span ,
    QuadTable quad          ,
):
    """ Evaluates a function on quadrature nodes of an interval.

    Parameters
    ----------
    fun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`
        Function to be evaluated.
    ndim : :class:`python:int`
        Number of output dimensions of the integrand.
    x_span : :class:`python:tuple` (:obj:`numpy:numpy.float64`, :obj:`numpy:numpy.float64`)
        Lower and upper bound of the evaluation interval.
    quad : :class:`QuadTable`
        Normalized evaluation nodes.

    Returns
    -------
    :class:`numpy:numpy.ndarray`:class:`(shape = (ndim), dtype = np.float64)`
        The approximated value of the integral.

    """  

    cdef Py_ssize_t nsteps = quad._w.shape[0]
    cdef Py_ssize_t isteps

    cdef ccallback_t callback
    ccallback_prepare(&callback, signatures, fun, CCALLBACK_DEFAULTS)

    cdef np.ndarray[double, ndim=2, mode="c"] funvals_np = np.empty((nsteps, ndim),dtype=np.float64)
    cdef double[:,::1] funvals = funvals_np

    cdef double dx = x_span[1] - x_span[0]
    cdef double xi

    with nogil:

        for istep in range(nsteps):

            xi = x_span[0] + dx * quad._x[istep]

            # f_res = f(xi)
            LowLevelFun_apply(callback, xi, funvals[istep,:])

    ccallback_release(&callback)

    return funvals_np

cpdef np.ndarray[double, ndim=1, mode="c"] IntegrateOnSegment(
    object fun              ,
    int ndim                ,
    (double, double) x_span ,
    QuadTable quad          ,
    Py_ssize_t nint = 1     ,
    bint DoEFT = True       ,
):
    r""" Computes an approximation of the integral of a function on a segment.

    Denoting :math:`a \eqdef \text{x_span}[0]`, :math:`b \eqdef \text{x_span}[1]` and :math:`\Delta \eqdef \text{x_span}[1]-\text{x_span}[0]`, the integral is first decomposed into ``nint`` smaller integrals as:

    .. math::
        \int_a^b \operatorname{fun}(x) \dd  x = \sum_{i = 0}^{\text{nint}-1} \int_{a + \frac{i}{\text{nint}} * \Delta}^{a + \frac{i+1}{\text{nint}} * \Delta}  \operatorname{fun}(x) \dd x 

    Each of the smaller integrals is then approximated using the :class:`QuadTable` ``quad`` (cf formula :eq:`QuadTable_int_approx`).

    The integrand can either be:

    * A Python :obj:`python:callable` taking a :obj:`numpy:numpy.float64` as its sole argument, and returning a :class:`numpy:numpy.ndarray`:class:`(shape=ndim, dtype=np.float64)`.
    * A :class:`scipy:scipy.LowLevelCallable` for performance-critical use cases.

    See Also
    --------
    * :ref:`sphx_glr__build_auto_examples_convergence_integration_on_segment.py` for a quick demonstration on smooth functions, and the role of compensated summation.
    * :ref:`sphx_glr__build_auto_examples_benchmarks_quad_integration_lowlevel_bench.py` for an in-depth example and comparison of the different types of function input.

    Example
    -------

    Let us compare the approximation and exact value of the `Wallis integral <https://en.wikipedia.org/wiki/Wallis%27_integrals>`_ of order 7.

    .. math::
        \int_0^{\frac{\pi}{2}} \operatorname{sin}^7(x) \dd x = \frac{16}{35}

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
    fun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`
        Function to be integrated.
    ndim : :class:`python:int`
        Number of output dimensions of the integrand.
    x_span : :class:`python:tuple` (:obj:`numpy:numpy.float64`, :obj:`numpy:numpy.float64`)
        Lower and upper bound of the integration interval.
    quad : :class:`QuadTable`
        Weights and nodes of the integration.
    nint : :class:`python:int`, optional
        Number of sub-intervals for the integration, by default ``1``.
    DoEFT : :class:`python:bool`, optional
        Whether to use an error-free transformation for summation, by default :data:`python:True`.

    Returns
    -------
    :class:`numpy:numpy.ndarray`:class:`(shape = (nsteps, ndim), dtype = np.float64)`
        The approximated value of the integral.
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
        f_eft_comp = <double *> malloc(sizeof(double)*ndim)
        memset(f_eft_comp, 0, sizeof(double)*ndim)

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

cpdef np.ndarray[double, ndim=2, mode="c"] InterpolateOnSegment(
    double[:,::1] funvals   ,
    double[::1] x           ,
    (double, double) x_span ,
    QuadTable quad          ,
    double eps = 1e-14      ,
):
    r""" Interpolates a function given its value on quadrature nodes of an interval.

    `Lagrange interpolation <https://en.wikipedia.org/wiki/Lagrange_polynomial>`_ of a function evaluated at quadrature nodes of an interval.
    
    Given the :math:`n` function values :math:`f_i`, the Lagrange interpolating polynomial is the unique polynomial :math:`L_f` of degree :math:`n-1` such that :math:`L_f(x_i)=f_i`.

    Parameters
    ----------
    funvals : :class:`numpy:numpy.ndarray`:class:`(shape = (nx, ndim), dtype = np.float64)`
        Function values to be interpolated. See also :func:`EvalOnNodes`.
    x : :class:`numpy:numpy.ndarray`:class:`(shape = (nx), dtype = np.float64)`
        Array of nodes where the interpolation should be evaluated.
    x_span : :class:`python:tuple` (:obj:`numpy:numpy.float64`, :obj:`numpy:numpy.float64`)
        Lower and upper bound of the evaluation interval.
    quad : :class:`QuadTable`
        Normalized evaluation nodes.
    eps : :obj:`numpy:numpy.float64`
        If :math:`|x-x_i|<\text{eps}`, then the approximation :math:`l_i(x)=1.` is used to avoid division by zero. By default ``1e-14``.

    Returns
    -------
    :class:`numpy:numpy.ndarray`:class:`(shape = (ndim), dtype = np.float64)`
        The approximated value of the integral.

    """  

    assert funvals.shape[0] == quad._w.shape[0]
    assert x_span[1] > x_span[0]

    cdef int ndim = funvals.shape[1]
    cdef int nx = x.shape[0]
    cdef int nsteps = quad._w.shape[0]

    cdef np.ndarray[double, ndim=2, mode="c"] res = np.empty((nx,ndim) ,dtype=np.float64)

    cdef double *l
    cdef double *xscal
    cdef int i
    cdef double dxinv

    with nogil:
        
        l = <double*> malloc(sizeof(double)*nx*nsteps)
        xscal = <double*> malloc(sizeof(double)*nx)
        
        dxinv = 1./(x_span[1] - x_span[0])

        for i in range(nx):
            xscal[i] = (x[i] - x_span[0]) * dxinv

        ComputeLagrangeWeights(
            xscal       ,
            nx          ,
            quad._wlag  , 
            quad._x     ,
            l           ,
            eps         ,
        )

        # res = l . funvals
        scipy.linalg.cython_blas.dgemm(transn,transn,&ndim,&nx,&nsteps,&one_double,&funvals[0,0],&ndim,l,&nsteps,&zero_double,&res[0,0],&ndim)

        free(xscal)
        free(l)

    return res

@cython.cdivision(True)
cdef inline void ComputeLagrangeWeights(
    double *xscal       ,
    int nx              ,
    double[::1] wlag    ,
    double[::1] x       ,
    double *l           ,
    double eps          ,
) noexcept nogil:

    cdef double* cur_l
    cdef double* cur_x_l
    cdef double* cur_xscal = xscal
    cdef double xscal_val

    cdef int ix
    cdef int min_istep = 0
    cdef int istep
    cdef int nsteps = wlag.shape[0]

    cdef double cursum
    cdef double absdx
    cdef double min_dx
    cdef double dx

    for ix in range(nx):

        min_dx = DBL_MAX

        cur_x_l = l + ix * nsteps
        cur_l = cur_x_l
        xscal_val = xscal[ix]

        for istep in range(nsteps):

            cur_l[0] = xscal_val - x[istep]
            absdx = cfabs(cur_l[0])
            
            if absdx < min_dx:
                min_dx = absdx
                min_istep = istep

            cur_l += 1

        if min_dx < eps:

            memset(cur_x_l, 0, sizeof(double)*nsteps)
            cur_x_l[min_istep] = 1.

        else:

            cursum = 0.
            cur_l = cur_x_l

            for istep in range(nsteps):

                cur_l[0] = wlag[istep] / cur_l[0] 
                cursum += cur_l[0]

                cur_l += 1

            cursum = 1./cursum
            scipy.linalg.cython_blas.dscal(&nsteps, &cursum, cur_x_l, &int_one)
