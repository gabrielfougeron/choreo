import numpy as np
import math as m
import scipy
import itertools

np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    linewidth = 150,
    floatmode = "fixed",
)

def proj_to_zero(array, eps=1e-14):
    for idx in itertools.product(*[range(i)  for i in array.shape]):
        if abs(array[idx]) < eps:
            array[idx] = 0.

        
print()        
print("="*80)
print()

# FFT of complex arrays

pint = 2
qint = 3
nint = pint * qint

f = np.random.random((nint)) + 1j * np.random.random((nint))

fft_f = scipy.fft.fft(f)

rf = f.copy().reshape(qint, pint) 

rf_1 = scipy.fft.fft(rf, axis=0) 

for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
        
# CAREFUL ! TRANSPOSE HERE !        
fft_rf = scipy.fft.fft(rf_1.T, axis=0).reshape(-1)

print(np.linalg.norm(fft_f - fft_rf))

print()        
print("="*80)
print()

# IDEM tranposed

pint = 2
qint = 3
nint = pint * qint

f = np.random.random((nint)) + 1j * np.random.random((nint))

fft_f = scipy.fft.fft(f)

rf = f.copy().reshape(qint, pint) 

# CAREFUL ! TRANSPOSE HERE !    
rf_1 = scipy.fft.fft(rf.T, axis=1)


for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[ip, iq] *= w
    
fft_rf = scipy.fft.fft(rf_1, axis=0).reshape(-1)


print(np.linalg.norm(fft_f - fft_rf))


print()        
print("="*80)
print()


# IDEM with array of transposed shape (since pint and qint have similar roles: nint = pint * qint = qint * pint)

pint = 2
qint = 3
nint = pint * qint

f = np.random.random((nint)) + 1j * np.random.random((nint))

fft_f = scipy.fft.fft(f)

rf = f.copy().reshape(pint,qint) 

# CAREFUL ! TRANSPOSE HERE !    
rf_1 = scipy.fft.fft(rf.T, axis=1)


for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
    
fft_rf = scipy.fft.fft(rf_1, axis=0).reshape(-1)


print(np.linalg.norm(fft_f - fft_rf))


print()        
print("="*80)
print()

# IDEM IFFT

ifft_f = scipy.fft.ifft(f)

rf = f.copy().reshape(qint, pint) 

rf_1 = scipy.fft.ifft(rf, axis=0)

for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
        
# CAREFUL ! TRANSPOSE HERE !        
ifft_rf = scipy.fft.ifft(rf_1, axis=1).T.reshape(-1)

print(np.linalg.norm(ifft_f - ifft_rf))



print()        
print("="*80)
print()

# FFT of real arrays


qint = 2 * 3
pint = 3
nint = pint * qint

ncoeff = nint // 2 + 1
qcoeff = qint // 2 + 1

f = np.random.random((nint))

rfft_f = scipy.fft.rfft(f)

rf = f.copy().reshape(qint, pint) 

rf_1 = scipy.fft.rfft(rf, axis=0)

for iq in range(qcoeff):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
        
 
rfft_rf_quad = scipy.fft.fft(rf_1, axis=1)

reorder = np.zeros((ncoeff),dtype=np.complex128)

# Weird reordering algorithm

j = 0
jq = 0
jp = 0
reorder[j] = rfft_rf_quad[jq,jp]
jqdir =  1
jpdir = -1
for ip in range(pint):   
    for iq in range(qcoeff-1):
        j = j + 1
        jq = jq + jqdir
        
        reorder[j] = rfft_rf_quad[jq,jp].real + 1j*jqdir*rfft_rf_quad[jq,jp].imag
        

    jqdir = - jqdir
    
    jp = (jp + jpdir + pint) % pint
    jpdir = - jpdir - jqdir
  
        
        
# print(rfft_f)
# print(rfft_rf_quad)
        
print(np.linalg.norm(rfft_f - reorder))


print()        
print("="*80)
print()


# IRFFT


# qint = 2 * 101
# pint = 37
# nint = pint * qint

ncoeff = nint // 2 + 1
qcoeff = qint // 2 + 1

f_sol = np.random.random((nint))
# f_sol = f.copy()

cf = scipy.fft.rfft(f_sol)

reorder = np.zeros((qcoeff,pint),dtype=np.complex128)

# Weird reordering algorithm

j = 0
jq = 0
jp = 0
reorder[jq,jp] = cf[j]
jqdir =  1
jpdir = -1
for ip in range(pint):   
    for iq in range(qcoeff-1):
        j = j + 1
        jq = jq + jqdir
        
        reorder[jq,jp] = cf[j].real + 1j*jqdir*cf[j].imag
        
    jqdir = - jqdir
    
    jp = (jp + jpdir + pint) % pint
    jpdir = - jpdir - jqdir
    
    reorder[jq,jp] = cf[j].real + 1j*jqdir*cf[j].imag


rfft_rf_quad = scipy.fft.ifft(reorder, axis=1)

for iq in range(qcoeff):
    for ip in range(pint):        
        
        w = np.exp((2j*m.pi*ip*iq)/nint)
        
        rfft_rf_quad[iq, ip] *= w
        
rfft_total = scipy.fft.irfft(rfft_rf_quad, axis=0).reshape(-1)
        

print(np.linalg.norm(f_sol - rfft_total))


print()        
print("="*80)
print()


# Now for the general setup, with complex numbers:


ncoeffs_min = 2
nparam_per_period = 3
nperiods = 3
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)



# Making the tables artificially longer before FFT + Naive convolution

All_params_basis_long = np.zeros((nint,nparam_per_period),dtype=np.complex128)
all_params_long = np.zeros((nparam_per_period,nint),dtype=np.complex128)

for iint in range(nint):

    All_params_basis_long[iint,: ] = All_params_basis[iint // nperiods,:]
        
for iint in range(nint):
        
    all_params_long[:, iint] = all_params[:, iint % nperiods]  
    
ifft_f_long = scipy.fft.ifft(All_params_basis_long, axis=0)
ifft_g_long = scipy.fft.ifft(all_params_long, axis=1) # This is sparse with sparsity pattern one in ncoeffs_min


convolve_ifft_fg = np.zeros((nint),dtype = np.complex128)

for iint in range(nint):
    for jint in range(nint):
        
        ijint = (iint - jint + nint) % nint
        
        convolve_ifft_fg[iint] += np.dot(ifft_f_long[jint,:], ifft_g_long[:,ijint])
        
print(np.linalg.norm(all_pos_direct - convolve_ifft_fg))
print()         



# Finding the right tables


ifft_f  = scipy.fft.ifft(All_params_basis, axis=0)
ifft_g  = scipy.fft.ifft(all_params, axis=1)


ifft_f_long_bis = np.zeros((nint,nparam_per_period),dtype=np.complex128)
ifft_g_long_bis = np.zeros((nparam_per_period,nint),dtype=np.complex128) 


for ip in range(nperiods):
    for iq in range(ncoeffs_min):
    
        jint = iq + ncoeffs_min*ip

        if jint == 0:
            mul = 1.
        else:

            mul = 1./nperiods * (1-np.exp(2j*m.pi*iq/ncoeffs_min)) / (1-np.exp(2j*m.pi*jint/nint))
            
        ifft_f_long_bis[jint,:] = ifft_f[iq,:] * mul


print(np.linalg.norm(ifft_f_long - ifft_f_long_bis))

print()

for ip in range(nperiods):
    ifft_g_long_bis[:,ip * ncoeffs_min] = ifft_g[:,ip]

print(np.linalg.norm(ifft_g_long - ifft_g_long_bis))

# Reduced convolution version I

convolve_ifft_fg = np.zeros((nint),dtype = np.complex128)

ifft_f_mod = ifft_f.copy()

for iq in range(ncoeffs_min):
    
    if iq == 0:
        mul = 1.
    else:        
        mul = (1-np.exp(2j*m.pi*iq / ncoeffs_min))

    ifft_f_mod[iq,:] *= mul


for ip in range(nperiods):
    
    for jpp in range(ip,ip+nperiods):
            
        for iq in range(ncoeffs_min):
                    
            jp = jpp-ip
            
            iint = ((jpp)*ncoeffs_min + iq) % nint
            
            jint = iq + ncoeffs_min * (jpp-ip)
            
            if iq == 0:
                if jp == 0:
                    mul = 1.
                else:
                    mul = 0.
            else:
                mul = 1. / (nperiods * (1-np.exp(2j*m.pi*jint / nint)))


            convolve_ifft_fg[iint] += np.dot(ifft_f_mod[iq,:], ifft_g[:,ip]) * mul
        

        
print()        
print(np.linalg.norm(all_pos_direct - convolve_ifft_fg))
#         

# Reduced convolution version II

convolve_ifft_fg = np.zeros((nint),dtype = np.complex128)

for ip in range(nperiods):
    for jint in range(nint):
        
        ijint = (ip * ncoeffs_min + jint) % nint

        convolve_ifft_fg[ijint] += np.dot(ifft_f_long[jint,:], ifft_g[:,ip])
        
        
print()        
print(np.linalg.norm(all_pos_direct - convolve_ifft_fg))



print()        
print("="*80)
print()


# Without convolution



ncoeffs_min = 2
nparam_per_period = 3
nperiods = 3
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)


ifft_f  = scipy.fft.ifft(All_params_basis, axis=0)

inter_array = np.zeros((ncoeffs_min,nperiods),dtype=np.complex128)

for iq in range(ncoeffs_min):
    
    for ip in range(nperiods):    
        
        w = np.exp((2j*m.pi*iq*ip)/nint)    
        
        inter_array[iq, ip] = w * np.matmul(ifft_f[iq,:], all_params[:,ip])
        

ifft_fg  = scipy.fft.ifft(inter_array, axis=1).T.reshape(-1)


print(np.linalg.norm(all_pos_direct - ifft_fg))


print()        
print("="*80)
print()


ncoeffs_min = 5
nparam_per_period = 15
nperiods = 11
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_g  = scipy.fft.ifft(all_params, axis=1)

inter_array = np.zeros((ncoeffs_min,nperiods),dtype=np.complex128)

for iq in range(ncoeffs_min):
    
    for ip in range(nperiods):        
        
        w = np.exp((2j*m.pi*iq*ip)/nint)
        
        inter_array[iq, ip] = w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])

ifft_fg  = scipy.fft.ifft(inter_array, axis=0).reshape(-1)
       
print(np.linalg.norm(all_pos_direct - ifft_fg))


print()        
print("="*80)
print()

# Without convolution, ON A SUBDOMAIN !!!!

ncoeffs_min = 2
nparam_per_period = 15
nperiods = 3
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_g  = scipy.fft.ifft(all_params, axis=1)

inter_array = np.zeros((ncoeffs_min,nperiods),dtype=np.complex128)

for iq in range(ncoeffs_min):
    
    for ip in range(nperiods):        
        
        w = np.exp((2j*m.pi*iq*ip)/nint)
        
        inter_array[iq, ip] = w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])

sum_fg = np.sum(inter_array, axis=0) / ncoeffs_min

ifft_fg  = scipy.fft.ifft(inter_array, axis=0).reshape(-1)
       
print(np.linalg.norm(all_pos_direct - ifft_fg))
print(np.linalg.norm(all_pos_direct[:nperiods] - sum_fg))

print()        
print("="*80)
print()

# Without convolution, on a subdomain, NO GLOBAL ARRAYS!!!!

ncoeffs_min = 3
nparam_per_period = 15
nperiods = 37
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)
all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_g  = scipy.fft.ifft(all_params, axis=1)

inter_array = np.zeros((nperiods),dtype=np.complex128)

ncoeffs_min_inv = 1.  / ncoeffs_min

for iq in range(ncoeffs_min):
    
    wo = np.exp((2j*m.pi*iq)/nint)
    
    w = ncoeffs_min_inv
    
    for ip in range(nperiods):        
        
        inter_array[ip] += w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])
        
        w *= wo

print(np.linalg.norm(all_pos_direct[:nperiods] - inter_array))

# IS THIS REALLY WHAT I WANT THOUGH ?
# DO THIS WITH ULTIMATELY REAL POSITIONS ?
# START WITH INVERSING THESE RELATIONS ? (<=> go from positions to parameters)


print()        
print("="*80)
print()

# IRFFT via ICFFT of double length

ncoeffs_min = 3
nparam_per_period = 15
nperiods = 5
nint = 2 * (ncoeffs_min * nperiods - 1)

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods))


""" This is probably actually not necessary
Making sure that the first and last coeffs are purely real. There might be a better way.
all_params[:,0] = 0.
all_params[:,-1] = 0.
 """
all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.irfft(all_coeffs)

assert nint == all_pos_direct.shape[0]

all_coeffs_2 = np.zeros((nint), dtype=np.complex128)

all_coeffs_2[:all_coeffs.shape[0]] = 2*all_coeffs
all_coeffs_2[0] = all_coeffs[0]
all_coeffs_2[all_coeffs.shape[0]-1] = all_coeffs[all_coeffs.shape[0]-1]

all_pos_2 = scipy.fft.ifft(all_coeffs_2)
all_pos_2_real = all_pos_2.real.copy()

print(np.linalg.norm(all_pos_direct - all_pos_2_real))

all_params_3 = np.zeros((nparam_per_period, 2*nperiods))
all_params_3[:,:nperiods] = all_params[:,:]
all_coeffs_3 = np.dot(All_params_basis, all_params_3).T.reshape(-1)[0:-2].copy()
all_coeffs_3[1:all_coeffs.shape[0]-1] *= 2


print(np.linalg.norm(all_coeffs_2 - all_coeffs_3))


print()        
print("="*80)
print()

# IRFFT via ICFFT of double length with decomposable nint

ncoeffs_min = 3
nparam_per_period = 15
nperiods = 5
nint = 2 * (ncoeffs_min * nperiods)

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods))


# This is probably actually not necessary
# Making sure that the first and last coeffs are purely real. There might be a better way.
# all_params[:,0] = 0.
# all_params[:,-1] = 0.

 
 
all_coeffs = np.zeros((nint//2+1), dtype=np.complex128)
all_coeffs[:nint//2] = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.irfft(all_coeffs)
assert nint == all_pos_direct.shape[0]

all_coeffs_2 = np.zeros((nint), dtype=np.complex128)

all_coeffs_2[:all_coeffs.shape[0]] = 2*all_coeffs
all_coeffs_2[0] = all_coeffs[0]
all_coeffs_2[all_coeffs.shape[0]-1] = all_coeffs[all_coeffs.shape[0]-1]

all_pos_2 = scipy.fft.ifft(all_coeffs_2)
all_pos_2_real = all_pos_2.real.copy()

print(np.linalg.norm(all_pos_direct - all_pos_2_real))

all_params_3 = np.zeros((nparam_per_period, 2*nperiods))
all_params_3[:,:nperiods] = all_params[:,:]
all_coeffs_3 = np.dot(All_params_basis, all_params_3).T.reshape(-1)
all_coeffs_3[1:all_coeffs.shape[0]-1] *= 2
print(np.linalg.norm(all_coeffs_2 - all_coeffs_3))






all_coeffs_4 = np.dot(All_params_basis, all_params_3).T.reshape(-1)
all_pos_4_direct = scipy.fft.ifft(all_coeffs_4)

# The average value is treated separately
meanval = -np.dot(All_params_basis[0,:],all_params_3[:,0]).real / nint
# All parameters are doubled here.
# The transform is a COMPLEX Inverse transform here
ifft_g  = scipy.fft.ifft(2*all_params_3, axis=1)

n_inter = nperiods
inter_array = np.full((n_inter), meanval)

ncoeffs_min_inv = 1.  / ncoeffs_min

for iq in range(ncoeffs_min):
    
    wo = np.exp((2j*m.pi*iq)/nint)
    
    w = ncoeffs_min_inv
    
    for ip in range(n_inter):        
        # Only the real part of the computation is needed here
        inter_array[ip] += (w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])).real
        
        w *= wo

# print(np.linalg.norm(all_pos_4_direct[:n_inter] - inter_array))
print(np.linalg.norm(all_pos_direct[:n_inter] - inter_array))
# print(all_pos_direct[:n_inter])
# print(inter_array)
# print(all_pos_direct[:n_inter] - inter_array)



print()        
print("="*80)
print()

# IRFFT via ICFFT of double length with decomposable nint and real transforms


ncoeffs_min = 7
nparam_per_period = 3
nperiods = 37
nint = 2 * (ncoeffs_min * nperiods)

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods))
 
all_coeffs = np.zeros((nint//2+1), dtype=np.complex128)
all_coeffs[:nint//2] = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.irfft(all_coeffs)
assert nint == all_pos_direct.shape[0]



# The average value is treated separately
meanval = -np.dot(All_params_basis[0,:].real, all_params[:,0]) / nint
# All parameters are doubled here.
# ifft_g  = scipy.fft.ihfft(all_params, axis=1, n=2*nperiods) # identical but ihfft is less often available in fft packages
ifft_g  = np.conjugate(scipy.fft.rfft(all_params, axis=1, n=2*nperiods))/(2*nperiods)

n_inter = nperiods + 1
inter_array = np.full((n_inter), meanval)

ncoeffs_min_inv = 2.  / ncoeffs_min

for iq in range(ncoeffs_min):
    
    wo = np.exp((2j*m.pi*iq)/nint)
    
    w = ncoeffs_min_inv
    
    for ip in range(n_inter):        
        # Only the real part of the computation is needed here
        inter_array[ip] += (w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])).real
        
        w *= wo

print(np.linalg.norm(all_pos_direct[:n_inter] - inter_array))




print()        
print("="*80)
print()

# standard vector convolution

nint = 7

a = np.random.random((nint)) + 1j * np.random.random((nint))
b = np.random.random((nint)) + 1j * np.random.random((nint))

c = np.zeros((nint), dtype=np.complex128)

for k in range(nint):
    for i in range(nint):
        ik = (nint + k - i ) % nint
        c[k] += a[i] * b[ik]


fft_a = scipy.fft.fft(a)
fft_b = scipy.fft.fft(b)
fft_c = scipy.fft.fft(c)

print(np.linalg.norm(fft_c - fft_a * fft_b))


c = a * b
ifft_a = scipy.fft.ifft(a)
ifft_b = scipy.fft.ifft(b)
ifft_c = scipy.fft.ifft(c)

convo = np.zeros((nint), dtype=np.complex128)

for k in range(nint):
    for i in range(nint):
        ik = (nint + k - i ) % nint
        convo[k] += ifft_a[i] * ifft_b[ik]


print(np.linalg.norm(convo - ifft_c))






print()        
print("="*80)
print()


# Reconstruction of signal of the form a * 1^T or 1 * b^T.

ncoeffs_min = 3
nperiods = 2
nint = ncoeffs_min * nperiods


All_params_basis = np.random.random((ncoeffs_min)) + 1j * np.random.random((ncoeffs_min))
all_params = np.random.random((nperiods)) + 1j * np.random.random((nperiods))


ifft_f = scipy.fft.ifft(All_params_basis)
ifft_g = scipy.fft.ifft(all_params)


All_params_basis_mat = np.matmul(All_params_basis.reshape((ncoeffs_min,1)),  np.ones((1,nperiods), dtype=np.complex128))
all_params_mat = np.matmul( np.ones((ncoeffs_min,1), dtype=np.complex128), all_params.reshape((1,nperiods)))


ifft_f_mat = scipy.fft.ifft(All_params_basis_mat.reshape(-1))
ifft_g_mat = scipy.fft.ifft(all_params_mat.reshape(-1))


ifft_g_recons = np.zeros((nint), dtype=np.complex128)
for kq in range(nperiods):
    ifft_g_recons[kq*ncoeffs_min] = ifft_g[kq]
    
print(np.linalg.norm(ifft_g_recons - ifft_g_mat))

ifft_f_recons = np.zeros((nint), dtype=np.complex128)

for j in range(nint):
    
    jq, jp = divmod(j,ncoeffs_min)
    
    if j == 0:
        
        fac = 1.
        
    else:
        
        wn = np.exp(2j*np.pi*j/ncoeffs_min)
        wd = np.exp(2j*np.pi*j/nint)
        
        fac = (1-wn)/(1-wd)/nperiods
    
    ifft_f_recons[j] = ifft_f[jp] * fac

print(np.linalg.norm(ifft_f_recons - ifft_f_mat))




print()        
print("="*80)
print()


# Using convolution



ncoeffs_min = 3
nparam_per_period = 1
nperiods = 6
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_f = scipy.fft.ifft(All_params_basis, axis=0)
ifft_g = scipy.fft.ifft(all_params, axis=1)
ifft_fg = np.matmul(ifft_f, ifft_g)


All_params_basis_mat = np.zeros((ncoeffs_min, nperiods, nparam_per_period), dtype=np.complex128)
for ip in range(nperiods):
    All_params_basis_mat[:,ip,:] = All_params_basis

ifft_f_mat = scipy.fft.ifft(All_params_basis_mat.reshape((nint,nparam_per_period)), axis=0).reshape((ncoeffs_min, nperiods, nparam_per_period))


convo = np.zeros((nint), dtype=np.complex128)

for k in range(nint):
    
    kp, kq = divmod(k, nperiods)
    
    for i in range(nint):
        ik = (nint + k - i ) % nint
        
        ip, iq = divmod(i, nperiods)
        iiq, iip = divmod(i, ncoeffs_min)
        ikp, ikq = divmod(ik, ncoeffs_min)
        
        if ikq == 0:

            if i == 0:
                
                fac = 1.
                
            else:
                
                wn = np.exp(2j*np.pi*i/ncoeffs_min)
                wd = np.exp(2j*np.pi*i/nint)
                
                fac = (1-wn)/(1-wd)/nperiods
                
                
            convo[k] += ifft_fg[iip, ikp] * fac


print(np.linalg.norm(convo - all_pos_direct))


# Another version

n_inter = nint
convo = np.zeros((n_inter), dtype=np.complex128)
 
for ikp in range(nperiods):
        
    for k in range(n_inter):

        ik = ikp*ncoeffs_min
        i = (nint + k - ik ) % nint
        iiq, iip = divmod(i, ncoeffs_min)

        if i == 0:
            fac = 1.
        else:
            wn = np.exp(2j*np.pi*i/ncoeffs_min)
            wd = np.exp(2j*np.pi*i/nint)
            fac = (1-wn)/(1-wd)/nperiods
            
        convo[k] += ifft_fg[iip, ikp] * fac


print(np.linalg.norm(convo - all_pos_direct[:n_inter]))


# Another version

convo = np.zeros((nint), dtype=np.complex128)
 
for ikp in range(nperiods):

    ik = ikp*ncoeffs_min

    for iip in range(ncoeffs_min):
        
        for iiq in range(nperiods):
            
            i = iip + ncoeffs_min*iiq
            
            k = (i + ik ) % nint
            
            if i == 0:
                fac = 1.
            else:
                wn = np.exp(2j*np.pi*iip/ncoeffs_min)
                wd = np.exp(2j*np.pi*i/nint)
                fac = (1-wn)/(1-wd)/nperiods
                
            convo[k] += ifft_fg[iip, ikp] * fac


print(np.linalg.norm(convo - all_pos_direct[:nint]))


# Another version

ninter = nint
convo = np.zeros((ninter), dtype=np.complex128)

for iip in range(ncoeffs_min):
         
    for ikp in range(nperiods): 

        ik = ikp*ncoeffs_min
        
        for iiq in range(nperiods):
            
            ii = (ikp + iiq) % nperiods
            k = iip + ii*ncoeffs_min

            i = iip + ncoeffs_min*iiq            
            if i == 0:
                fac = 1.
            else:
                wn = np.exp(2j*np.pi*iip/ncoeffs_min)
                wd = np.exp(2j*np.pi*i/nint)
                fac = (1-wn)/(1-wd)/nperiods
                
            convo[k] += ifft_fg[iip, ikp] * fac


print(np.linalg.norm(convo - all_pos_direct[:ninter]))



# Another version
ninter = 1
convo = np.zeros((ninter), dtype=np.complex128)
 
for ikp in range(nperiods):
    
    ik = ikp*ncoeffs_min
        
    for k in range(ninter):
        
        iip = k % ncoeffs_min

        i = (nint + k - ik ) % nint

        if i == 0:
            fac = 1.
        else:
            wn = np.exp(2j*np.pi*i/ncoeffs_min)
            wd = np.exp(2j*np.pi*i/nint)
            fac = (1-wn)/(1-wd)/nperiods
            
            
        convo[k] += ifft_fg[iip, ikp] * fac


print(np.linalg.norm(convo - all_pos_direct[:ninter]))



print()        
print("="*80)
print()


# No convolution ?

ncoeffs_min = 7
nparam_per_period = 3
nperiods = 5
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_f = scipy.fft.ifft(All_params_basis, axis=0)


all_pos = np.zeros((nint), dtype=np.complex128)

woo = np.exp(2j*np.pi/nint)
wo = 1.
for ip in range(ncoeffs_min):

    all_params_cp = all_params.copy()
    
    w = 1.
    for iq in range(nperiods):
        all_params_cp[:,iq] *= w
        w *= wo
    wo *= woo
    
    ifft_g = scipy.fft.ifft(all_params_cp, axis=1)

    all_pos[ip::ncoeffs_min] = np.matmul(ifft_f[ip,:], ifft_g)


print(np.linalg.norm(all_pos_direct - all_pos))


# No copy

ncoeffs_min = 2
nparam_per_period = 3
nperiods = 2
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_f = scipy.fft.ifft(All_params_basis, axis=0)


all_pos_mat = np.zeros((nperiods, ncoeffs_min), dtype=np.complex128)

wo = np.exp(2j*np.pi/nint)
for ip in range(ncoeffs_min):

    if (ip != 0):
        w = 1.
        for iq in range(nperiods):
            all_params[:,iq] *= w
            w *= wo
    
    ifft_g = scipy.fft.ifft(all_params, axis=1)

    all_pos_mat[:,ip] = np.matmul(ifft_f[ip,:], ifft_g)

all_pos = all_pos_mat.reshape(-1)

print(np.linalg.norm(all_pos_direct - all_pos))