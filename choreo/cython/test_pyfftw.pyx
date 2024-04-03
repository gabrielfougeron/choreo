cimport pyfftw
from libc.stdlib cimport malloc, free

cdef class mystruct:

    cdef pyfftw.fftw_exe** exe_arr

    def pyfftw_list_to_carr(self, pyfftw_list):

        cdef Py_ssize_t nfft = len(pyfftw_list)
        cdef Py_ssize_t i

        cdef pyfftw.FFTW fftw_object
        self.exe_arr = <pyfftw.fftw_exe**> malloc(sizeof(pyfftw.fftw_exe*) * nfft)

        for i in range(nfft):
            fftw_object = pyfftw_list[i]
            self.exe_arr[i] =  <pyfftw.fftw_exe*> malloc(sizeof(pyfftw.fftw_exe))
            self.exe_arr[i][0] = fftw_object.get_fftw_exe()


    


cpdef void object_execute_in_nogil(pyfftw.FFTW fftw_object):

    with nogil:
        fftw_object.execute_nogil()

