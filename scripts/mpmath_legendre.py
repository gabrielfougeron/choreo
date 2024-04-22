import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import numpy as np
import choreo 
import mpmath
import pyquickbench

dps = 30
mpmath.mp.dps = dps

n = 10

rk = choreo.scipy_plus.multiprec_tables.ComputeGaussButcherTables(n,method='Lobatto_IIIC')

