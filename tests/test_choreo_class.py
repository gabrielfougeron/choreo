import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions


__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

a_list = ['a','b','c','d']
not_a_list_ = 123

test_action = choreo.ChoreoAction(a_list=a_list,not_a_list_=not_a_list_,current_cvg_lvl=0)

print(test_action.a())