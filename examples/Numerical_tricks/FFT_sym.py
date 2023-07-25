"""
FFT and symmetries
==================

A few numerical investigation of how symmetries of a signal determine the shape of its discrete Fourier transform, and how to take advantage of this fact to optimize performance.


"""
# %%
# The spectral solvers in choreo make extensive use of the discrete Fourier transform to go back end forth between time :math:`\sin` and frequency representation of orbits.
#
# .. math::
#    c_k = \sum_{j=-n}^{n} x_j \exp(-2i\pi k\tfrac{j}{n} )



# Note that the total number of nodes needs to be divisible by 4 to properly account for symmetries

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

n_base = 100  # Base number of nodes. This could be any integer > 0
n = 4 * n_base   # Total number of nodes
# %%
# Signals we use in choreo are aways real-valued. This implies that the FFT has Hermitian symmetry, meaning that the 

# %%

x = np.array(range(n))+1
plt.plot(x)
plt.show()

# %%
# LaTeX syntax in the text blocks does not require backslashes to be escaped:
#
# .. math::
#    \sin


# 
# rfft_c = scipy.fft.rfft(x)
# # print(rfft_c.size)
# 
# 
# print('')
# print('')
# print('')
# print('Even function => DCT I')
# 
# # y = np.random.random(n//2+1)
# y = np.array(range(n//2+1))+1
# x = np.zeros(n)
# for i in range(n//2+1):
#     x[i] = y[i]
# for i in range(n//2+1,n):
#     x[i] = y[n - i]
# 
# 
# plt.plot(x)
# plt.savefig("even.png")
# plt.close()
# 
# 
# rfft_c = scipy.fft.rfft(x)
# # print(rfft_c.real)
# print("Norm of imaginary part of rfft")
# print( np.linalg.norm(rfft_c.imag))
# 
# dct_I_c = scipy.fft.dct(x[0:n//2+1],1)
# idct_I_x = scipy.fft.idct(dct_I_c,1)
# 
# # print(dct_I_c)
# print("Norm difference Btw Real part of RFFT and DCT I")
# print( np.linalg.norm(rfft_c.real - dct_I_c))
# 
# # print(dct_I_c)
# print("Norm difference Btw initial and round trip DCT I")
# print( np.linalg.norm(x[0:n//2+1] - idct_I_x))
# 
# 
# 
# 
# 
# 
# print('')
# print('')
# print('')
# print('Odd function ==> DST I')
# 
# # y = np.random.random(n//2-1)
# y = np.array(range(n//2-1))+1
# x = np.zeros(n)
# for i in range(1,n//2):
#     x[i] = y[i-1]
# for i in range(n//2+1,n):
#     x[i] = - y[n - i - 1]
# 
# plt.plot(x)
# plt.savefig("odd.png")
# plt.close()
# 
# rfft_c = scipy.fft.rfft(x)
# print("Norm of real part of rfft")
# print( np.linalg.norm(rfft_c.real))
# 
# 
# dst_I_c = scipy.fft.dst(x[1:n//2],1)
# idst_I_x = scipy.fft.idst(dst_I_c,1)
# 
# # print(-rfft_c.imag[1:n//2])
# # print(dst_I_c)
# 
# # # print(dst_I_c)
# print("Norm difference Btw Real part of RFFT and DST I")
# print( np.linalg.norm(rfft_c.imag[1:n//2] - (- dst_I_c)))
# 
# print("Norm difference Btw initial and round trip  DST I")
# print( np.linalg.norm(x[1:n//2] - idst_I_x))
# 
# 
# 
# 
# # 
# print('')
# print('')
# print('')
# print('Even function + Odd quarter point => DCT III')
# 
# # y = np.random.random(n//4)
# y = np.array(range(n//4))+1
# x = np.zeros(n)
# for i in range(n//4):
#     # print(i,i)
#     x[i] = y[i]
# # print('quarter')
# for i in range(n//4+1,n//2):
#     # print(i,n//2 - i)
#     x[i] = - y[n//2 - i]
# # print("midway")
# for i in range(n//2,n//2 + n//4):
#     # print(i,i-n//2)
#     x[i] = - y[i-n//2]
# # print("three quarters")
# for i in range(n//2 + n//4 + 1,n):
#     # print(i,n - i)
#     x[i] = y[n - i]
# 
# 
# plt.plot(x)
# plt.savefig("even - odd.png")
# plt.close()
# 
# 
# rfft_c = scipy.fft.rfft(x)
# 
# print("Norm of imaginary part of rfft")
# print( np.linalg.norm(rfft_c.imag))
# print("Norm of half the coeffs of rfft")
# print( np.linalg.norm(rfft_c[::2]))
# 
# dct_III_c = scipy.fft.dct(x[0:n//4],3)
# 
# # print(dct_III_c.size)
# # print(rfft_c.real[1::2])
# # print(dct_III_c*2)
# 
# idct_III_c = scipy.fft.idct(dct_III_c,3)
# 
# # print(dct_c)
# print("Norm difference Btw Real part of RFFT and DCT III")
# print( np.linalg.norm(rfft_c.real[1::2] - 2*dct_III_c))
# 
# # print(dct_c)
# print("Norm difference Btw initial and round trip DCT")
# print( np.linalg.norm(x[0:n//4] - idct_III_c))
# 
# 
# 
# 
# 
# 
# 
# print('')
# print('')
# print('')
# print('Odd function + Even quarter point => DST III')
# 
# # y = np.random.random(n//4)
# y = np.array(range(n//4))+1
# x = np.zeros(n)
# for i in range(1,n//4 + 1):
#     # print(i,i-1)
#     x[i] = y[i-1]
# # print('quarter')
# for i in range(n//4+1,n//2):
#     # print(i,n//2-1 - i)
#     x[i] = y[n//2-1 - i]
# # print("midway")
# for i in range(n//2+1,n//2 + n//4+1):
#     # print(i,i-(n//2+1))
#     x[i] = - y[i-(n//2+1)]
# # print("three quarters")
# for i in range(n//2 + n//4+1,n):
#     # print(i,n-1 - i)
#     x[i] = -y[n-1 - i]
# 
# 
# plt.plot(x)
# plt.savefig("odd - even.png")
# plt.close()
# 
# 
# rfft_c = scipy.fft.rfft(x)
# 
# print("Norm of real part of rfft")
# print( np.linalg.norm(rfft_c.real))
# print("Norm of half the coeffs of rfft")
# print( np.linalg.norm(rfft_c[::2]))
# 
# dst_III_c = scipy.fft.dst(x[1:n//4+1],3)
# 
# # print(dct_III_c.size)
# # print(rfft_c.imag[1::2])
# # print(dst_III_c*2)
# 
# idst_III_c = scipy.fft.idst(dst_III_c,3)
# 
# print("Norm difference Btw Real part of RFFT and DST III")
# print( np.linalg.norm(rfft_c.imag[1::2] - (-2*dst_III_c)))
# 
# print("Norm difference Btw initial and round trip DST III")
# print( np.linalg.norm(x[1:n//4+1] - idst_III_c))
# 
# 
# 
# 
# 
# 
# 
# 
# 
# print('')
# print('')
# print('')
# print('Function with subperiod => strided RFFT')
# 
# 
# 
# n_sub_period = 5
# n_base = n
# 
# n = n_base * n_sub_period
# 
# 
# y = np.random.random(n_base)
# x = np.zeros(n)
# 
# for i in range(n_sub_period):
#     for j in range(n_base):
#         x[j+i*n_base] = y[j]
# 
# 
# rfft_c = scipy.fft.rfft(x)
# # print(rfft_c.real)
# 
# print("Norm of coeffs c_k where k != i * n_sub_period")
# err = 0
# for i in range(1,n_sub_period):
#     err += np.linalg.norm(rfft_c[i::n_sub_period])
# 
# print(err)
# 
# 
# rfftsub_c = scipy.fft.rfft(x[:n_base])
# 
# 
# print("Norm difference Btw strided RFFT and sub RFFT")
# print( np.linalg.norm(rfft_c[::n_sub_period] - n_sub_period*rfftsub_c))
# 
# 
# 
