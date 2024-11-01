
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np

import scipy
import slycot
import sys
import random


# np.set_printoptions(precision=3,suppress=True,linewidth=np.nan,threshold=sys.maxsize)


__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)


import choreo

n = 10


w = np.zeros((2*n,2*n))
w[0:n,n:2*n] =  np.identity(n)
w[n:2*n,0:n] = -np.identity(n)
w = np.ascontiguousarray(w)

def find_n(M):

    (p,q) = M.shape
    assert p == q

    rem = p % 2
    assert rem == 0

    return p // 2

def hamil_err(M):

    nn = find_n(M)

    a_err = np.linalg.norm(M[0:nn   ,0:nn   ] + M[n:2*nn    ,n:2*nn ].transpose())
    b_err = np.linalg.norm(M[n:2*nn ,0:nn   ] - M[n:2*nn    ,0:nn   ].transpose())
    c_err = np.linalg.norm(M[0:nn   ,n:2*nn ] - M[0:nn      ,n:2*nn ].transpose())

    return 2*a_err + b_err + c_err

def sk_hamil_err(M):

    nn = find_n(M)

    a_err = np.linalg.norm(M[0:nn   ,0:nn   ] - M[n:2*nn    ,n:2*nn ].transpose())
    b_err = np.linalg.norm(M[n:2*nn ,0:nn   ] + M[n:2*nn    ,0:nn   ].transpose())
    c_err = np.linalg.norm(M[0:nn   ,n:2*nn ] + M[0:nn      ,n:2*nn ].transpose())

    return 2*a_err + b_err + c_err

def sym_err(M):
    return np.linalg.norm(M-M.transpose())

def sk_sym_err(M):
    return np.linalg.norm(M+M.transpose())

def ortho_err(M):
    (p,q) = M.shape
    assert p == q

    return np.linalg.norm(np.dot(M,M.transpose())-np.identity(p))

def symplect_err(M):
    nn = find_n(M)
    ww = np.zeros((2*nn,2*nn))
    ww[0:nn,nn:2*nn] =  np.identity(nn)
    ww[nn:2*nn,0:nn] = -np.identity(nn)
    ww = np.ascontiguousarray(ww)

    return np.linalg.norm(np.dot(M.transpose(),np.dot(ww,M))-ww)


def cpt_perm_mat(perm):

    nn = perm.size
    perm_mat = np.zeros((nn,nn))
    check = np.zeros(nn,dtype=int)

    for i in range(nn):

        perm_mat[i,perm[i]] = 1
        check[perm[i]] = 1

    assert sum(check) == nn

    return perm_mat

def make_fit_perm(W):

    nn = find_n(W)

    bot = []
    top = []

    for i in range(nn):

        ii = 2*i
        jj = 2*i+1

        if (W[ii,jj] > 0):
            bot.append(ii)
            top.append(jj)
        else:
            bot.append(jj)
            top.append(ii)

    l = bot
    l.extend(top)

    return np.array(l)

def OneItSqrt(InputMat,AutoMat,AutoMatInv):
    
    return (InputMat + np.dot(AutoMatInv,np.dot(InputMat,AutoMat))) / 2

def OneHighamIt1(A,Yk,Zk):

    Ykp = (Yk + scipy.linalg.solve(Yk,A)) / 2
    Zkp = (Zk + scipy.linalg.solve(Zk.transpose(),A.transpose()).transpose()) / 2

    return Ykp , Zkp

def OneHighamIt2(Pk,Qk):

    Pkp = (Pk + scipy.linalg.inv(Qk)) / 2
    Qkp = (Qk + scipy.linalg.inv(Pk)) / 2

    return Pkp , Qkp


mat = np.random.random_sample((2*n,2*n))

hmat = (np.dot(w,np.dot(mat.transpose(),w)) + mat) / 2
skhmat = (np.dot(w,np.dot(mat.transpose(),w)) - mat) / 2

print(hamil_err(hmat))
print(sk_hamil_err(skhmat))


hmat2 = np.dot(hmat,hmat)
print(sk_hamil_err(hmat2))

print('')
print('hmat2_sqrt')

hmat2_sqrt = scipy.linalg.sqrtm(hmat2)
print(hamil_err(hmat2))
print(sk_hamil_err(hmat2))
print(np.linalg.norm(hmat - hmat2_sqrt))
print(np.linalg.norm(np.dot(hmat2_sqrt,hmat2_sqrt) - hmat2))


print('')
print('skhmat_sqrt')

skhmat_sqrt = scipy.linalg.sqrtm(skhmat)
print(hamil_err(skhmat_sqrt))
print(sk_hamil_err(skhmat_sqrt))
print(np.linalg.norm(np.dot(skhmat_sqrt,skhmat_sqrt) - skhmat))


print('')
print('sk_sym')

# skhmat = np.dot(hmat,hmat)

sk_sym = np.dot(w,skhmat)
print(sk_sym_err(sk_sym))




sk_sym_schur_T, sk_sym_schur_Z = scipy.linalg.schur(sk_sym)


print(np.linalg.norm(np.dot(w,skhmat) - np.dot(sk_sym_schur_Z,np.dot(sk_sym_schur_T,sk_sym_schur_Z.transpose()))))


print(sk_sym_err(sk_sym_schur_T))


# perm = [2*i for i in range(n)]
# perm.extend([2*i+1 for i in range(n)])
# perm = np.array(perm)


perm = make_fit_perm(sk_sym_schur_T)

perm_mat = cpt_perm_mat(perm)


print(ortho_err(perm_mat))

Delta = np.dot(perm_mat,np.dot(sk_sym_schur_T,perm_mat.transpose()))
Q     =                 np.dot(sk_sym_schur_Z,perm_mat.transpose())



print(ortho_err(Q))



print(hamil_err(Delta))
print(sk_sym_err(Delta))
# print(abs(Delta)> 1e-10)
# 
print(np.linalg.norm(np.dot(w,skhmat) - np.dot(Q,np.dot(Delta,Q.transpose()))))


# print(np.dot(Q,np.dot(Delta,Q.transpose())))
# print(np.dot(w,skhmat))
# print(np.dot(w,skhmat) - np.dot(Q,np.dot(Delta,Q.transpose())))

D_vec = np.diag(Delta,n)
print(np.all(D_vec > 0))
d_vec = np.sqrt(D_vec)

Beta = np.diag(np.append(d_vec,d_vec))

print(np.linalg.norm(Delta - np.dot(Beta,np.dot(w,Beta))))


print(hamil_err(Delta))
print(sk_sym_err(Delta))

# QwQT wH - wH QwQT
A = np.dot(Q,np.dot(w,Q.transpose()))
np.linalg.norm(np.dot(A,sk_sym) - np.dot(sk_sym,A))

# 
print('')
print('tests')

print('QwQT')
A = np.dot(Q,np.dot(w,Q.transpose()))



print(sym_err(A))
print(sk_sym_err(A))
print(hamil_err(A))
print(sk_hamil_err(A))
print(ortho_err(A))
print(symplect_err(A))

# B = np.dot(A,sk_sym) - np.dot(sk_sym,A)
# print(np.linalg.norm(B))

# print('QwQT')
# A = np.dot(Q,np.dot(w,np.dot(Q.transpose(),w)))
# 
# 
# print(sym_err(A))
# print(sk_sym_err(A))
# print(hamil_err(A))
# print(sk_hamil_err(A))
# print(ortho_err(A))
# print(symplect_err(A))


# lie = np.dot(skhmat,hmat) - np.dot(hmat,skhmat)

# print(sym_err(lie))
# print(sk_sym_err(lie))
# print(hamil_err(lie))
# print(sk_hamil_err(lie))
# print(ortho_err(lie))
# print(symplect_err(lie))

# print(np.dot(A,sk_sym))
# 

# 
print('')
print('Iterative algo')


nit = 20

# sqrtmat = (np.identity(2*n) +  np.copy(mat))/2

# A = skhmat
A = hmat


P = np.copy(A)
Q = np.identity(2*n)


for i in range(nit):


    P,Q = OneHighamIt2(P,Q)



print(np.linalg.norm(np.dot(P,P)-A))
print(np.linalg.norm(np.dot(Q,Q)-scipy.linalg.inv(A)))
print('')
print(sym_err(P))
print(sk_sym_err(P))
print(hamil_err(P))
print(sk_hamil_err(P))
print(ortho_err(P))
print(symplect_err(P))
print('')
print(sym_err(Q))
print(sk_sym_err(Q))
print(hamil_err(Q))
print(sk_hamil_err(Q))
print(ortho_err(Q))
print(symplect_err(Q))

# 
# 
# print('')
# print('slycot')
# 
# routine_list = [
#     'mb03xd']
# 
# for routine in routine_list:
#     print(f'{routine} is in slycot : {routine in dir(slycot)}')



choreo.SymplecticSchurOfSkewHamilton(skhmat)
# choreo.Hamiltonian_sqrt_of_skew_Hamiltonian(np.dot(hmat,hmat))

eps = 1e-12
all_success = True
ntests = 10000

for i in range(ntests):

    sign_a = 1 if (random.random() > 1/2) else -1
    sign_b = 1 if (random.random() > 1/2) else -1

    a =   sign_a * random.random()
    b =   sign_b * random.random()
    c = - sign_b * random.random()

    mat = np.array([[a,b],[c,a]])

    alpha, beta, gamma = choreo.sqrt_2x2_mat(a,b,c)

    mat_sqrt = np.array([[alpha,beta],[gamma,alpha]])

    err = np.linalg.norm(mat-np.dot(mat_sqrt,mat_sqrt))

    success = (err < eps)

    all_success = all_success and (err < eps)

    if not(success):

        print(err)
# 
#         print(mat)
#         print(np.dot(mat_sqrt,mat_sqrt))





print(all_success)
