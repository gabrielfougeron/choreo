'''
linalg.py : Define linear algebra related things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m
import scipy.linalg

def null_space(A, eps = 1e-12):
    # Why would this be better than scipy.linalg.null_space ???

    n = A.shape[0]
    m = A.shape[1]

    if (n == 0):

        return np.identity(m)

    else:

        Q, R, P = scipy.linalg.qr(
            A.T,
            overwrite_a = False ,
            mode = 'full'       ,
            # mode = 'economic'       ,
            pivoting = True     ,
        )
        
        # assert np.linalg.norm(A[P,:] - np.matmul(R.T, Q.T)) < eps 
        
        k = min(R.shape)

        for i in range(k):
            if (abs(R[i, i]) < eps):
                rank = i
                break
        else:
            rank = min(R.shape)

        # nullspace_dim = m - rank
        # nullspace = np.zeros((m, nullspace_dim), dtype=np.float64)

        return np.ascontiguousarray(Q[:, rank:])

        # return scipy.linalg.null_space(A, rcond=eps)


def random_orthogonal_matrix(geodim):

    mat = np.random.random_sample((geodim,geodim))
    sksymmat = mat - mat.T
    return scipy.linalg.expm(sksymmat)

def InstabilityDecomposition(Mat, eps=1e-12):
    """ Order and pairs up the complex eivenvalues / eigenspaces of a real matrix.
    """

    n,m = Mat.shape
    assert n==m

    eigvals, eigvects = scipy.linalg.eig(a=Mat, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)

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
            
    assert i == n 

    return Instability_magnitude, Instability_directions

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
