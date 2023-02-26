import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy
import functools

# import choreo

#####
import mkl_fft._numpy_fft
the_rfft  = mkl_fft._numpy_fft.rfft
the_irfft = mkl_fft._numpy_fft.irfft

######
# import scipy.fft
# the_rfft = scipy.fft.rfft
# the_irfft = scipy.fft.irfft

# ######
# the_rfft = np.fft.rfft
# the_irfft = np.fft.irfft




ncoeff = 100
nint = 2*ncoeff



all_pos = np.random.random(nint)
c_coeffs = the_rfft(all_pos,norm="forward")
all_posp = the_irfft(c_coeffs,n=nint,norm="forward")

err = np.linalg.norm(all_pos - all_posp)
print(err)


c_coeffs = np.random.random(ncoeff) + 1j * np.random.random(ncoeff)
c_coeffs[0] = c_coeffs[0].real

all_pos = the_irfft(c_coeffs,n=nint,norm="forward")
c_coeffsp = the_rfft(all_pos,norm="forward")[0:ncoeff]


err = np.linalg.norm(c_coeffs - c_coeffsp)
print(err)



