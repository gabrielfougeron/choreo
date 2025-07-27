from libc.math cimport fabs as cfabs

import numpy as np
cimport numpy as np
np.import_array()
cimport cython

# arr is assumed contiguous
cpdef proj_to_zero(arr, double eps = 1e-12):

    cdef double[::1] arr_flat = arr.reshape(-1)
    cdef Py_ssize_t size = arr_flat.shape[0]
    cdef Py_ssize_t i

    for i in range(size):
        if cfabs(arr_flat[i]) < eps:
            arr_flat[i] = 0.
        
cpdef void inplace_transpose(double[:,::1] arr):

    assert arr.shape[0] == arr.shape[1]

    cdef Py_ssize_t i,j
    cdef double tmp

    for i in range(1,arr.shape[0]-1):
        for j in range(i+1,arr.shape[0]):

            tmp = arr[i,j]
            arr[i,j] = arr[j,i]
            arr[j,i] = tmp