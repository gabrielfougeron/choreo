'''
Choreo_scipy_plus.py : Define things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m
import scipy.optimize
import scipy.linalg as la
import scipy.sparse as sp
import functools

from choreo.Choreo_cython_scipy_plus import ExplicitSymplecticWithTable_XV_cython
from choreo.Choreo_cython_scipy_plus import ExplicitSymplecticWithTable_VX_cython
from choreo.Choreo_cython_scipy_plus import SymplecticStormerVerlet_XV_cython
from choreo.Choreo_cython_scipy_plus import SymplecticStormerVerlet_VX_cython

from choreo.Choreo_scipy_plus_nonlin import nonlin_solve_pp

class current_best:
    # Class meant to store the best solution during scipy optimization / root finding
    # Useful since scipy does not return the best solution, but rather the solution at the last iteration.
    
    def __init__(self,x,f):
        
        self.x = x
        self.f = f
        self.f_norm = np.linalg.norm(f)
        
    def update(self,x,f,f_norm):

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






#####################
# EXPLICIT RK STUFF #
#####################


c_table_Euler = np.array([1.])
d_table_Euler = np.array([1.])
assert c_table_Euler.size == d_table_Euler.size
nsteps_Euler = c_table_Euler.size
SymplecticEuler_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)
SymplecticEuler_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Euler,d_table=d_table_Euler,nsteps=nsteps_Euler)

c_table_Ruth3 = np.array([1.        ,-2./3  ,2/3    ])
d_table_Ruth3 = np.array([-1./24    , 3./4  ,7./24  ])
assert c_table_Ruth3.size == d_table_Ruth3.size
nsteps_Ruth3 = c_table_Ruth3.size
SymplecticRuth3_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)
SymplecticRuth3_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Ruth3,d_table=d_table_Ruth3,nsteps=nsteps_Ruth3)

curt2 = m.pow(2,1./3)
c_table_Ruth4 = np.array([1./(2*(2-curt2))  ,(1-curt2)/(2*(2-curt2))    ,(1-curt2)/(2*(2-curt2))    ,1./(2*(2-curt2))   ])
d_table_Ruth4 = np.array([1./(2-curt2)      ,-curt2/(2-curt2)           ,1./(2-curt2)               ,0.                 ])
assert c_table_Ruth4.size == d_table_Ruth4.size
nsteps_Ruth4 = c_table_Ruth4.size
SymplecticRuth4_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)
SymplecticRuth4_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Ruth4,d_table=d_table_Ruth4,nsteps=nsteps_Ruth4)

c_table_Ruth4Rat = np.array([0.     , 1./3  , -1./3     , 1.        , -1./3 , 1./3  ])
d_table_Ruth4Rat = np.array([7./48  , 3./8  , -1./48    , -1./48    ,  3./8 , 7./48 ])
assert c_table_Ruth4Rat.size == d_table_Ruth4Rat.size
nsteps_Ruth4Rat = c_table_Ruth4Rat.size
SymplecticRuth4Rat_XV = functools.partial(ExplicitSymplecticWithTable_XV_cython,c_table=c_table_Ruth4Rat,d_table=d_table_Ruth4Rat,nsteps=nsteps_Ruth4Rat)
SymplecticRuth4Rat_VX = functools.partial(ExplicitSymplecticWithTable_VX_cython,c_table=c_table_Ruth4Rat,d_table=d_table_Ruth4Rat,nsteps=nsteps_Ruth4Rat)

all_SymplecticIntegrators = {
    'SymplecticEuler'               : SymplecticEuler_XV,
    'SymplecticEuler_XV'            : SymplecticEuler_XV,
    'SymplecticEuler_VX'            : SymplecticEuler_VX,
    'SymplecticStormerVerlet'       : SymplecticStormerVerlet_XV_cython,
    'SymplecticStormerVerlet_XV'    : SymplecticStormerVerlet_XV_cython,
    'SymplecticStormerVerlet_VX'    : SymplecticStormerVerlet_VX_cython,
    'SymplecticRuth3'               : SymplecticRuth3_XV,
    'SymplecticRuth3_XV'            : SymplecticRuth3_XV,
    'SymplecticRuth3_VX'            : SymplecticRuth3_VX,
    'SymplecticRuth4'               : SymplecticRuth4_XV,
    'SymplecticRuth4_XV'            : SymplecticRuth4_XV,
    'SymplecticRuth4_VX'            : SymplecticRuth4_VX,
    'SymplecticRuth4Rat'            : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_XV'         : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_VX'         : SymplecticRuth4Rat_VX,
}

all_unique_SymplecticIntegrators = {
    'SymplecticEuler_XV'            : SymplecticEuler_XV,
    'SymplecticEuler_VX'            : SymplecticEuler_VX,
    'SymplecticStormerVerlet_XV'    : SymplecticStormerVerlet_XV_cython,
    'SymplecticStormerVerlet_VX'    : SymplecticStormerVerlet_VX_cython,
    'SymplecticRuth3_XV'            : SymplecticRuth3_XV,
    'SymplecticRuth3_VX'            : SymplecticRuth3_VX,
    'SymplecticRuth4_XV'            : SymplecticRuth4_XV,
    'SymplecticRuth4_VX'            : SymplecticRuth4_VX,
    'SymplecticRuth4Rat_XV'         : SymplecticRuth4Rat_XV,
    'SymplecticRuth4Rat_VX'         : SymplecticRuth4Rat_VX,
}

def GetSymplecticIntegrator(method='SymplecticRuth3'):

    return all_SymplecticIntegrators[method]




#####################
# IMPLICIT RK STUFF #
#####################

a_table_Gauss_1 = np.array([[1.]])
b_table_Gauss_1 = np.array([1.])
c_table_Gauss_1 = np.array([1.])
nsteps_Gauss_1 = a_table_Gauss_1.shape[0]
assert a_table_Gauss_1.shape[1] == nsteps_Gauss_1
assert b_table_Gauss_1.size == nsteps_Gauss_1
assert c_table_Gauss_1.size == nsteps_Gauss_1

db_G2 = m.pow(3,-1/2)/2
a_table_Gauss_2 = np.array([[ 1./4  , 1./4 - db_G2  ],[ 1./4 + db_G2    , 1./4   ]])
b_table_Gauss_2 = np.array([  1./2  , 1./2 ])
c_table_Gauss_2 = np.array([  1./2 - db_G2 , 1./2 + db_G2 ])
nsteps_Gauss_2 = a_table_Gauss_2.shape[0]
assert a_table_Gauss_2.shape[1] == nsteps_Gauss_2
assert b_table_Gauss_2.size == nsteps_Gauss_2
assert c_table_Gauss_2.size == nsteps_Gauss_2





def InstabilityDecomposition(Mat,eps=1e-12):

    n,m = Mat.shape
    assert n==m

    eigvals,eigvects = scipy.linalg.eig(a=Mat, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)

    idx_sort = np.argsort(-abs(eigvals))
    Instability_magnitude = abs(eigvals)[idx_sort]
    
    Instability_directions = np.zeros((n,n))

    i = 0
    while (i < n):

        is_real = (np.linalg.norm(eigvects[:,idx_sort[i]].imag) < eps) and (abs(eigvals[idx_sort[i  ]].imag) < eps)

        if is_real :

            Instability_directions[i,:] = eigvects[:,idx_sort[i]].real

            i += 1
            
        else :

            assert (i+1) < n

            is_conj_couple = ((np.linalg.norm(eigvects[:,idx_sort[i]].imag + eigvects[:,idx_sort[i+1]].imag)) < eps) and (abs(eigvals[idx_sort[i  ]].imag + eigvals[idx_sort[i+1]].imag) < eps)    

            assert is_conj_couple

            Instability_directions[i  ,:] = eigvects[:,idx_sort[i]].real
            Instability_directions[i+1,:] = eigvects[:,idx_sort[i]].imag

            i += 2

    return Instability_magnitude,Instability_directions

def algo_H(n,k,z):

    w = np.zeros((n))

    sig = 0.
    for i in range(k,n):
        sig += z[i]*z[i]
    sig = m.sqrt(sig)

    if z[k] > 0 :
        w[k] = z[k] + sig
    else:
        w[k] = z[k] - sig

    for i in range(k+1,n):
        w[i] = z[i]

    return w
        
def apply_H_sym(n,A,w):
    # Computes H A HT
    
    Aw = np.dot(A,w)
    wTA = np.dot(w,A)
    wTAw = np.dot(Aw,w)
    wTw = np.dot(w,w)

    two_ovr_wTw = 2/wTw
    alpha = wTAw * two_ovr_wTw

    for i in range(n):
        for j in range(n):

            A[i,j] += two_ovr_wTw * (alpha * w[i] * w[j] - Aw[i] * w[j]  - w[i] * wTA[j])

def apply_H_left(n,A,w):
    # Computes H A

    wTA = np.dot(w,A)
    wTw = np.dot(w,w)

    two_ovr_wTw = 2/wTw

    for i in range(n):
        for j in range(n):

            A[i,j] -= two_ovr_wTw *  w[i] * wTA[j]

def algo_J(n,k,y,z):

    sig = np.hypot(y[k],z[k])

    if sig < 1e-14:
        return 1., 0.
    else:
        return y[k]/sig , z[k]/sig
        
def apply_J_sym(n,k,c,s,D,U,V):
    # Computes J A J^T

    C = np.identity(n)
    S = np.zeros((n,n))

    C[k,k] = c
    S[k,k] = s

    CDC = C @ D @ C
    SDC = S @ D @ C
    CDS = C @ D @ S 
    SDS = S @ D @ S 

    CVC = C @ V @ C
    SVC = S @ V @ C
    # CVS = C @ V @ S 
    SVS = S @ V @ S 

    CUC = C @ U @ C
    # SUC = S @ U @ C
    CUS = C @ U @ S 
    SUS = S @ U @ S 
    
    D[:,:] =   CDC + SVC + CUS + SDS.T
    U[:,:] = - CDS - SVS + CUC + CDS.T
    V[:,:] = - SDC + CVC - SUS + SDC.T
            
def apply_J_left(n,k,c,s,Q,P):
    # Computes J A

    C = np.identity(n)
    S = np.zeros((n,n))

    C[k,k] = c
    S[k,k] = s

    QQ = C @ Q - S @ P
    PP = C @ P + S @ Q

    Q[:,:] = QQ
    P[:,:] = PP

def sqrt_2x2_mat(a,b,c):
    #                                         [ a  b ]
    # Computes real square root of 2x2 matrix [ c  a ]
    # where b*c < 0
    
    bc = b*c
    delta = a*a - bc
    p = (a + m.sqrt(delta))/2
    q = bc / (4*p)

    sign =  1
    # sign = -1

    alpha = sign * m.sqrt(p)

    if b > 0:
        beta =   sign * m.sqrt(q*b/c)
    else:
        beta = - sign * m.sqrt(q*b/c)

    gamma = q / beta

    return alpha,beta,gamma

def SymplecticSchurOfSkewHamilton(H):

    (p,q) = H.shape
    assert p == q

    rem = p % 2
    assert rem == 0

    n = p // 2

    A = np.copy(H[0:n   ,0:n    ])
    B = np.copy(H[0:n   ,n:2*n  ])
    C = np.copy(H[n:2*n ,0:n    ])

    eps = 1e-12

    assert np.linalg.norm(A - H[n:2*n,n:2*n].transpose()) < eps
    assert np.linalg.norm(B + B.transpose()) < eps
    assert np.linalg.norm(C + C.transpose()) < eps

    Q = np.identity(n)
    P = np.zeros((n,n))

    # Reduction to upper Hessenberg

    for k in range(n-2):

        w = algo_H(n,k+1,C[:,k])

        apply_H_sym(n,A,w)
        apply_H_sym(n,B,w)
        apply_H_sym(n,C,w)

        apply_H_left(n,Q,w)
        apply_H_left(n,P,w)

        c, s = algo_J(n,k+1,A[:,k],C[:,k])

        apply_J_sym(n,k+1,c,s,A,B,C)
        apply_J_left(n,k+1,c,s,Q,P)

        w = algo_H(n,k+1,A[:,k])

        apply_H_sym(n,A,w)
        apply_H_sym(n,B,w)
        apply_H_sym(n,C,w)

        apply_H_left(n,Q,w)
        apply_H_left(n,P,w)


    k = n-2
    c, s = algo_J(n,k+1,A[:,k],C[:,k])
    apply_J_sym(n,k+1,c,s,A,B,C)
    apply_J_left(n,k+1,c,s,Q,P)

    T,Z = scipy.linalg.schur(A)

    #     [  Q  P ] [ T   W  ] [  Q  P ] T
    # H = [ -P  Q ] [ 0  T^T ] [ -P  Q ] 

    Q = Z.T @ Q
    P = Z.T @ P


    QP = np.zeros((2*n,2*n))

    QP[0:n   ,0:n    ] =  Q
    QP[0:n   ,n:2*n  ] =  P
    QP[n:2*n ,0:n    ] = -P
    QP[n:2*n ,n:2*n  ] =  Q

    # print(abs(QP @ H @ QP.T) < eps)
    print(QP @ H @ QP.T)
    

    #     M = scipy.linalg.sqrtm(A)
    #     N = scipy.linalg.solve_sylvester(M, -M.T, B)
    # 
    #     print(M)
    #     print(N)
