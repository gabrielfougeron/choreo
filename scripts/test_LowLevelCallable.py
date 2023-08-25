import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import choreo 
import scipy

def py_fun(x):
    return x

cy_fun = scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.CallableInterface, "cy_fun")



res = choreo.scipy_plus.cython.CallableInterface.add_zero_and_one(py_fun)
print(res)
res = choreo.scipy_plus.cython.CallableInterface.add_zero_and_one(cy_fun)
print(res)


