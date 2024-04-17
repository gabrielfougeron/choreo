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


# deg = 7

for deg in range(2,10):
    print()
    print(deg)
    n = 3*2**(deg-1)

    TT = pyquickbench.TimeTrain()

    quad = choreo.scipy_plus.multiprec_tables.ComputeQuadrature(n, dps=dps, method="Gauss")

    # print(quad)

    TT.toc("Choreo")

    GL = mpmath.calculus.quadrature.GaussLegendre(mpmath.mp)

    res = GL.calc_nodes(deg, mpmath.mp.prec)

    TT.toc("mpmath")

    # print(TT)

    d = TT.to_dict(names_reduction=np.sum)
    print("time factor", d['Choreo']/d['mpmath'])

    for key, val in d.items():
        
        print(key, val/(n*n))
    
    tot_err = 0.
    for i in range(n//2):

        tot_err += (res[2*i+1][0]+1)/2 - quad.x[i]
        tot_err += res[2*i][1]/2 - quad.w[i]

    assert tot_err < 1e-16



# choreo.scipy_plus.multiprec_tables.ComputeGaussButcherTables(n,dps=dps,method="Gauss")