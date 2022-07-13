'''
Choreo_scipy_plus.py : Define things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m
import scipy.optimize
import scipy.linalg as la
import scipy.sparse as sp
import functools

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
    
def SymplecticEuler_Xfirst(fun,gun,t_span,x0,v0,nint):

    '''
    dx/dt = f(t,v)
    dv/dt = g(t,v)
    
    2 version. cf Wikipedia : https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    '''

    t = t_span[0]
    dt = (t_span[1] - t_span[0]) / nint

    x = x0
    v = v0

    for iint in range(nint):
 
        v = v + dt * gun(t,x)
        x = x + dt * fun(t,v)
        t += dt


    return x,v    

def SymplecticEuler_Vfirst(fun,gun,t_span,x0,v0,nint):

    '''
    dx/dt = f(t,v)
    dv/dt = g(t,v)
    
    2 version. cf Wikipedia : https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    '''

    t = t_span[0]
    dt = (t_span[1] - t_span[0]) / nint

    x = x0
    v = v0

    for iint in range(nint):

        x = x + dt * fun(t,v)
        t += dt
        v = v + dt * gun(t,x)


    return x,v

SymplecticEuler = SymplecticEuler_Xfirst

def SymplecticStormerVerlet_XV(fun,gun,t_span,x0,v0,nint):

    t = t_span[0]
    dt = (t_span[1] - t_span[0]) / nint

    x = x0
    v = v0

    v = v + dt/2 * gun(t,x)

    for iint in range(nint-1):
 
        x = x + dt * fun(t,v)
        t += dt

        v = v + dt * gun(t,x)

    x = x + dt * fun(t,v)
    t += dt
    v = v + dt/2 * gun(t,x)

    return x,v

def SymplecticStormerVerlet_VX(fun,gun,t_span,x0,v0,nint):

    t = t_span[0]
    dt = (t_span[1] - t_span[0]) / nint

    x = x0
    v = v0

    x = x + dt/2 * fun(t,v)
    t += dt/2

    for iint in range(nint-1):

        v = v + dt * gun(t,x)

        x = x + dt * fun(t,v)
        t += dt

    v = v + dt * gun(t,x)
    
    x = x + dt/2 * fun(t,v)
    t += dt/2

    return x,v

SymplecticStormerVerlet = SymplecticStormerVerlet_VX

def SymplecticWithTable_VX(fun,gun,t_span,x0,v0,nint,c_table=None,d_table=None,nsteps=None):

    t = t_span[0]
    dt = (t_span[1] - t_span[0]) / nint

    x = x0
    v = v0

    for iint in range(nint):

        for istep in range(nsteps):
            
            x = x + (c_table[istep]*dt) * fun(t,v)
            t += c_table[istep]*dt
            v = v + (d_table[istep]*dt) * gun(t,x)

    return x,v

def SymplecticWithTable_XV(fun,gun,t_span,x0,v0,nint,c_table=None,d_table=None,nsteps=None):

    t = t_span[0]
    dt = (t_span[1] - t_span[0]) / nint

    x = x0
    v = v0

    for iint in range(nint):

        for istep in range(nsteps):

            v = v + (c_table[istep]*dt) * gun(t,x)            
            x = x + (d_table[istep]*dt) * fun(t,v)
            t += d_table[istep]*dt

    return x,v

c_table_Euler = np.array([1])
d_table_Euler = np.array([1])
assert c_table_Euler.size == d_table_Euler.size
nsteps_Euler = c_table_Euler.size
SymplecticEuler_Table_XV = functools.partial(SymplecticWithTable_XV,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)
SymplecticEuler_Table_VX = functools.partial(SymplecticWithTable_VX,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)

c_table_StormerVerlet = np.array([0.    ,1.      ])
d_table_StormerVerlet = np.array([1./2  ,1./2    ])
assert c_table_StormerVerlet.size == d_table_StormerVerlet.size
nsteps_StormerVerlet = c_table_StormerVerlet.size
SymplecticStormerVerlet_Table_XV = functools.partial(SymplecticWithTable_XV,c_table=c_table_StormerVerlet,d_table=d_table_StormerVerlet,nsteps=nsteps_StormerVerlet)
SymplecticStormerVerlet_Table_VX = functools.partial(SymplecticWithTable_VX,c_table=c_table_StormerVerlet,d_table=d_table_StormerVerlet,nsteps=nsteps_StormerVerlet)

c_table_Ruth3 = np.array([1.        ,-2./3  ,2/3    ])
d_table_Ruth3 = np.array([-1./24    , 3./4  ,7./24  ])
assert c_table_Ruth3.size == d_table_Ruth3.size
nsteps_Ruth3 = c_table_Ruth3.size
SymplecticRuth3_Table_XV = functools.partial(SymplecticWithTable_XV,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)
SymplecticRuth3_Table_VX = functools.partial(SymplecticWithTable_VX,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)

curt2 = m.pow(2,1./3)
c_table_Ruth4 = np.array([1./(2*(2-curt2))  ,(1-curt2)/(2*(2-curt2))    ,(1-curt2)/(2*(2-curt2))    ,1./(2*(2-curt2))   ])
d_table_Ruth4 = np.array([1./(2-curt2)      ,-curt2/(2-curt2)           ,1./(2-curt2)               ,0.                 ])
assert c_table_Ruth4.size == d_table_Ruth4.size
nsteps_Ruth4 = c_table_Ruth4.size
SymplecticRuth4_Table_XV = functools.partial(SymplecticWithTable_XV,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)
SymplecticRuth4_Table_VX = functools.partial(SymplecticWithTable_VX,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)

all_SymplecticIntegrators = {
    'SymplecticEuler' : SymplecticEuler,
    'SymplecticEuler_Xfirst' : SymplecticEuler_Xfirst,
    'SymplecticEuler_Vfirst' : SymplecticEuler_Vfirst,
    'SymplecticStormerVerlet' : SymplecticStormerVerlet,
    'SymplecticStormerVerlet_XV' : SymplecticStormerVerlet_XV,
    'SymplecticStormerVerlet_VX' : SymplecticStormerVerlet_VX,
    'SymplecticEuler_Table_XV' : SymplecticEuler_Table_XV,
    'SymplecticEuler_Table_VX' : SymplecticEuler_Table_VX,
    'SymplecticStormerVerlet_Table_XV' : SymplecticStormerVerlet_Table_XV,
    'SymplecticStormerVerlet_Table_VX' : SymplecticStormerVerlet_Table_VX,
    'SymplecticRuth3_Table_XV' : SymplecticRuth3_Table_XV,
    'SymplecticRuth3_Table_VX' : SymplecticRuth3_Table_VX,
    'SymplecticRuth4_Table_XV' : SymplecticRuth4_Table_XV,
    'SymplecticRuth4_Table_VX' : SymplecticRuth4_Table_VX,
    }

all_unique_SymplecticIntegrators = {
    'SymplecticEuler_Xfirst' : SymplecticEuler_Xfirst,
    'SymplecticEuler_Vfirst' : SymplecticEuler_Vfirst,
    'SymplecticStormerVerlet_XV' : SymplecticStormerVerlet_XV,
    'SymplecticStormerVerlet_VX' : SymplecticStormerVerlet_VX,
    'SymplecticEuler_Table_XV' : SymplecticEuler_Table_XV,
    'SymplecticEuler_Table_VX' : SymplecticEuler_Table_VX,
    'SymplecticStormerVerlet_Table_XV' : SymplecticStormerVerlet_Table_XV,
    'SymplecticStormerVerlet_Table_VX' : SymplecticStormerVerlet_Table_VX,
    'SymplecticRuth3_Table_XV' : SymplecticRuth3_Table_XV,
    'SymplecticRuth3_Table_VX' : SymplecticRuth3_Table_VX,
    'SymplecticRuth4_Table_XV' : SymplecticRuth4_Table_XV,
    'SymplecticRuth4_Table_VX' : SymplecticRuth4_Table_VX,
    }

def GetSymplecticIntegrator(method):

    return all_SymplecticIntegrators[method]








