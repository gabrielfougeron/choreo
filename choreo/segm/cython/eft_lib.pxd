cdef (double, double) Fast2Sum(double a, double b) noexcept nogil
cdef (double, double) TwoSum(double a, double b) noexcept nogil
cdef (double, double) Split(double a) noexcept nogil
cdef (double, double) TwoProduct(double a, double b) noexcept nogil
cdef void TwoSum_incr(double *y, double *d, double *e, int n) noexcept nogil
cdef void TwoSumScal_incr(double *y, double *d, double s, double *e, int n) noexcept nogil
cdef void FastVecSum(double* p, double* q, Py_ssize_t n) noexcept nogil
cdef void VecSum(double* p, double* q, Py_ssize_t n) noexcept nogil
cpdef double SumK(double[::1] v, Py_ssize_t k = *) noexcept
cpdef double FastSumK(double[::1] v, Py_ssize_t k = *)
cpdef void compute_r_vec(double[::1] v, double[::1] w, double[::1] r) noexcept nogil
cpdef double naive_dot(double[::1] v, double[::1] w)
cpdef double DotK(double[::1] v, double[::1] w, Py_ssize_t k = *)