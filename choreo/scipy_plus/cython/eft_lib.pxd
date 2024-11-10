cdef (double, double) Fast2Sum(double a, double b) noexcept nogil
cdef (double, double) TwoSum(double a, double b) noexcept nogil
cdef void TwoSum_incr(double *y, double *d, double *e, int n) noexcept nogil
cdef void FastVecSum(double* p, double* q, Py_ssize_t n) noexcept nogil
cdef void VecSum(double* p, double* q, Py_ssize_t n) noexcept nogil
cpdef double SumK(double[::1] v, Py_ssize_t k = *) noexcept
cpdef double FastSumK(double[::1] v, Py_ssize_t k = *)
cpdef double naive_sum_vect(double[:] v) noexcept nogil