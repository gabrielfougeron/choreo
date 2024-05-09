"""
Mock PyFFTW that does nothing
"""

cimport numpy as np

cdef class FFTW:

    cdef fftw_exe get_fftw_exe(self):

        cdef fftw_exe exe

        exe._fftw_execute = NULL
        exe._plan = NULL
        exe._input_pointer = NULL
        exe._output_pointer = NULL

        return exe

    cpdef update_arrays(self,
            new_input_array, new_output_array):
        pass

    cdef _update_arrays(self,
            np.ndarray new_input_array, np.ndarray new_output_array):
        pass

    cpdef execute(self):
        pass

    cdef void execute_nogil(self) noexcept nogil:
        pass

cdef void execute_in_nogil(fftw_exe* exe_ptr) noexcept nogil:
    pass