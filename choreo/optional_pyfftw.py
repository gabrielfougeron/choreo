from choreo.cython.optional_pyfftw import PYFFTW_AVAILABLE_COMPILE

try:
    assert PYFFTW_AVAILABLE_COMPILE
    import pyfftw as p_pyfftw   
    PYFFTW_AVAILABLE = True
except:
    import choreo.cython.pyfftw_fake as p_pyfftw   
    PYFFTW_AVAILABLE = False
