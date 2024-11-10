cdef extern from *:
    '''
    /* Defines complex types that are bit compatible with C99's complex.h
    * and (crucially) the same type as expected by fftw3.h.
    * Note, don't use this with complex.h. fftw3.h checks to see whether
    * complex.h is included and then uses that to set the interface.
    * Since MSVC doesn't support C99, by using the following types we
    * have a cross platform/compiler solution.
    *
    * */

    #ifndef CHOREO_COMPLEX_H
    #define CHOREO_COMPLEX_H

    typedef float cfloat[2];
    typedef double cdouble[2];
    typedef long double clongdouble[2];

    #endif /* Header guard */
    '''

    ctypedef float cfloat[2]
    ctypedef double cdouble[2]
    ctypedef long double clongdouble[2]

cdef char *transn
cdef char *transt

cdef int int_zero
cdef int int_one
cdef int int_two
cdef int int_minusone

cdef double minusone_double
cdef double half_double
cdef double two_double
cdef double one_double
cdef double zero_double
cdef double ctwopi
cdef double cminustwopi
cdef double ctwopisqrt2
cdef double cfourpi
cdef double ctwopisq
cdef double cfourpisq
