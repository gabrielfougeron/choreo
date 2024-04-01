import numpy as np
cimport numpy as np
np.import_array()

cimport pyfftw

cpdef void object_execute_in_nogil(pyfftw.FFTW fftw_object):

    with nogil:
        fftw_object.execute_nogil()