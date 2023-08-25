'''
ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !


'''

import os
cimport scipy.linalg.cython_blas
cimport scipy.linalg.cython_lapack

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

from .ccallback cimport (ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS,
                         ccallback_signature_t)



sigs = [
    (b"double (double)", 0),
    # (b"double (double, double, int *, void *)", 1)
]

# cdef ccallback_signature_t signatures[7]
cdef ccallback_signature_t signatures[2]

for idx, sig in enumerate(sigs):
    signatures[idx].signature = sig[0]
    signatures[idx].value = sig[1]

signatures[idx + 1].signature = NULL






cdef double cdef_cy_fun(double x) nogil:
    return x

cpdef double cpdef_cy_fun(double x) nogil:
    return x

def def_cy_fun(double x):
    return x








cdef double add_values_ann(
        void *data,
        long maxval,
    ) nogil:

    cdef ccallback_t *callback = <ccallback_t *>data
    cdef double result
    cdef double add = 0.0
    cdef double arg

    cdef long i
    
    for i in range(maxval):

        arg = i

        if callback.c_function != NULL:
            if callback.signature.value == 0:
                result = (<double(*)(double) nogil>callback.c_function)(arg)
        else:
            with gil:
                result = float((<object>callback.py_function)(arg))


        add = add + result

    return add


def add_values(
        callback_obj,
        long maxval = 10
    ):
    """
    Implementation of a caller routine in Cython
    """
    
    cdef ccallback_t callback
    cdef int error_flag = 0
    cdef double result

    ccallback_prepare(&callback, signatures, callback_obj, CCALLBACK_DEFAULTS)

    with nogil:
        result = add_values_ann(<void *>&callback, maxval)

    ccallback_release(&callback)

    return result


