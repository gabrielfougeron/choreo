import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import numpy as np
import choreo 
import mpmath
import pyquickbench
import chaospy

dps = 300
mpmath.mp.dps = dps

n = 30

# quad = choreo.scipy_plus.multiprec_tables.ComputeQuadrature(n, dps=dps, method='Cheb_I')
# print(quad)
# 
# quad = choreo.scipy_plus.multiprec_tables.ComputeQuadrature(n, dps=dps, method='Cheb_II')
# print(quad)


z = choreo.scipy_plus.multiprec_tables.ComputeQuadNodes(n, method = "ClenshawCurtis")
vdm_inv = choreo.scipy_plus.multiprec_tables.ComputeVandermondeInverseParker(n, z)
rhs = choreo.scipy_plus.multiprec_tables.Build_integration_RHS(z, n)
w = rhs * vdm_inv

quad = chaospy.quadrature.clenshaw_curtis(n-1)

for i in range(n):
    
    print(abs(z[i] - quad[0][0][i]))
    # print(quad[0][0][i])

# 
# print(quad[0])
# print(quad[1])

