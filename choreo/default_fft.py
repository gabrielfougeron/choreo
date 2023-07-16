'''
default_fft.py : Chooses default FFT functions, given availability
'''

try:

    import mkl_fft._numpy_fft

    default_rfft  = mkl_fft._numpy_fft.rfft
    default_irfft = mkl_fft._numpy_fft.irfft

except ImportError:

    try:

        import scipy.fft

        default_rfft = scipy.fft.rfft
        default_irfft = scipy.fft.irfft

    except ImportError:

        import numpy as np

        default_rfft = np.fft.rfft
        default_irfft = np.fft.irfft
