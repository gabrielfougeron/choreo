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








cdef double cy_fun(double x):
    return x


cdef double add_zero_and_one_ann(
        int *error_flag,
        void *data
    ) except? -1.0 nogil:

    cdef:
        ccallback_t *callback = <ccallback_t *>data
        double result
        double add = 0.0
        double arg

        int i
    

    for i in range(2):

        arg = i

        if callback.c_function != NULL:
            if callback.signature.value == 0:

                result = (<double(*)(double) nogil>callback.c_function)(arg)

        else:
            with gil:
                try:
                    result = float((<object>callback.py_function)(arg))
                except:  # noqa: E722
                    error_flag[0] = 1
                    raise

        add = add + result

    return add





def add_zero_and_one(callback_obj):
    """
    Implementation of a caller routine in Cython
    """
    cdef:
        ccallback_t callback
        int error_flag = 0
        double result

    ccallback_prepare(&callback, signatures, callback_obj, CCALLBACK_DEFAULTS)

    with nogil:
        result = add_zero_and_one_ann(&error_flag, <void *>&callback)

    ccallback_release(&callback)

    return result



#     
# 
# cdef double test_thunk_cython(double a, int *error_flag, void *data) except? -1.0 nogil:
#     """
#     Implementation of a thunk routine in Cython
#     """
#     cdef:
#         ccallback_t *callback = <ccallback_t *>data
#         double result = 0
# 
#     if callback.c_function != NULL:
#         if callback.signature.value == 0:
#             result = (<double(*)(double, int *, void *) nogil>callback.c_function)(
#                 a, error_flag, callback.user_data)
#         else:
#             result = (<double(*)(double, double, int *, void *) nogil>callback.c_function)(
#                 a, 0.0, error_flag, callback.user_data)
# 
#         if error_flag[0]:
#             # Python error has been set by the callback
#             return -1.0
#     else:
#         with gil:
#             try:
#                 return float((<object>callback.py_function)(a))
#             except:  # noqa: E722
#                 error_flag[0] = 1
#                 raise
# 
#     return result