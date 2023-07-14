import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import concurrent.futures
# import multiprocessing
import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions
import functools
import line_profiler

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


n = 2

a,b,c,beta,gamma = choreo.ComputeGaussButcherTables_np(n)

print(a)
print(b)
print(c)