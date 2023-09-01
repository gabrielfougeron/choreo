import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import numpy as np
import scipy
import choreo



@ProbabilisticTest()
def test_easy_sum():

    a = np.random.random()
    b = np.random.random()
    
    c = a+b
    print(c)
    
    choreo.scipy_plus.cython.eft_lib.add(a,b)






