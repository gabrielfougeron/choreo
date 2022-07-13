'''
Choreo_scipy_plus.py : Define things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m
import scipy.optimize
import scipy.linalg as la
import scipy.sparse as sp
import functools

from choreo.Choreo_cython_scipy_plus import SymplecticWithTable_XV_cython
from choreo.Choreo_cython_scipy_plus import SymplecticWithTable_VX_cython
from choreo.Choreo_cython_scipy_plus import SymplecticStormerVerlet_XV_cython
from choreo.Choreo_cython_scipy_plus import SymplecticStormerVerlet_VX_cython

class current_best:
    # Class meant to store the best solution during scipy optimization / root finding
    # Useful since scipy does not return the best solution, but rather the solution at the last iteration.
    
    def __init__(self,x,f):
        
        self.x = x
        self.f = f
        self.f_norm = np.linalg.norm(f)
        
    def update(self,x,f):
        
        f_norm = np.linalg.norm(f)
        
        if (f_norm < self.f_norm):
            self.x = x
            self.f = f
            self.f_norm = f_norm

    def get_best(self):
        return self.x,self.f,self.f_norm

class ExactKrylovJacobian(scipy.optimize.nonlin.KrylovJacobian):

    def __init__(self,exactgrad, rdiff=None, method='lgmres', inner_maxiter=20,inner_M=None, outer_k=10, **kw):

        scipy.optimize.nonlin.KrylovJacobian.__init__(self, rdiff, method, inner_maxiter,inner_M, outer_k, **kw)
        self.exactgrad = exactgrad

    def matvec(self, v):
        return self.exactgrad(self.x0,v)

    def rmatvec(self, v):
        return self.exactgrad(self.x0,v)

c_table_Euler = np.array([1.])
d_table_Euler = np.array([1.])
assert c_table_Euler.size == d_table_Euler.size
nsteps_Euler = c_table_Euler.size
SymplecticEuler_XV = functools.partial(SymplecticWithTable_XV_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)
SymplecticEuler_VX = functools.partial(SymplecticWithTable_VX_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)

c_table_Ruth3 = np.array([1.        ,-2./3  ,2/3    ])
d_table_Ruth3 = np.array([-1./24    , 3./4  ,7./24  ])
assert c_table_Ruth3.size == d_table_Ruth3.size
nsteps_Ruth3 = c_table_Ruth3.size
SymplecticRuth3_XV = functools.partial(SymplecticWithTable_XV_cython,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)
SymplecticRuth3_VX = functools.partial(SymplecticWithTable_VX_cython,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)

curt2 = m.pow(2,1./3)
c_table_Ruth4 = np.array([1./(2*(2-curt2))  ,(1-curt2)/(2*(2-curt2))    ,(1-curt2)/(2*(2-curt2))    ,1./(2*(2-curt2))   ])
d_table_Ruth4 = np.array([1./(2-curt2)      ,-curt2/(2-curt2)           ,1./(2-curt2)               ,0.                 ])
assert c_table_Ruth4.size == d_table_Ruth4.size
nsteps_Ruth4 = c_table_Ruth4.size
SymplecticRuth4_XV = functools.partial(SymplecticWithTable_XV_cython,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)
SymplecticRuth4_VX = functools.partial(SymplecticWithTable_VX_cython,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)

all_SymplecticIntegrators = {
    'SymplecticEuler' : SymplecticEuler_XV,
    'SymplecticEuler_XV' : SymplecticEuler_XV,
    'SymplecticEuler_VX' : SymplecticEuler_VX,
    'SymplecticStormerVerlet' : SymplecticStormerVerlet_XV_cython,
    'SymplecticStormerVerlet_XV' : SymplecticStormerVerlet_XV_cython,
    'SymplecticStormerVerlet_VX' : SymplecticStormerVerlet_VX_cython,
    'SymplecticRuth3_XV' : SymplecticRuth3_XV,
    'SymplecticRuth3_VX' : SymplecticRuth3_VX,
    'SymplecticRuth4_XV' : SymplecticRuth4_XV,
    'SymplecticRuth4_VX' : SymplecticRuth4_VX,
    }

def GetSymplecticIntegrator(method):

    return all_SymplecticIntegrators[method]








