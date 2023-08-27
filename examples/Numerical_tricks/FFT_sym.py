"""
FFT and symmetries
==================

A few numerical investigation of how symmetries of a signal determine the shape of its discrete Fourier transform, and how to take advantage of this fact to optimize performance.


"""
# %%
# The spectral solvers in choreo make extensive use of the discrete Fourier transform to go back end forth between time representation (positions :math:`x_j = x(j/n)`) and frequency representation (Fourier coefficients :math:`c_k`) representation of orbits. Explicit formulae read:
#
# .. math::
#    c_k = \sum_{j=0}^{n-1} x_j \exp\left(-2i\pi k\tfrac{j}{n} \right) \\
#    x_k = \frac{1}{n}\sum_{j=0}^{n-1} c_j \exp\left(2i\pi k\tfrac{j}{n} \right) \\
#
#
# The magic of FFT algorithms enables computation of **all** the positions from **all** the coefficients (and vice-versa) in :math:`\mathcal{O}(n\log(n))` time instead of the naive  :math:`\mathcal{O}(n^2)`.


# %%
# Note that the total number of nodes needs to be divisible by 4 to properly account for symmetries
import numpy as np
import scipy
import matplotlib.pyplot as plt

n_base = 8  # Base number of nodes. This could be any **even** integer > 0

# %%
# Real-valued signals and RFFT
# ****************************
#
# Signals we use in choreo are coordinates of the orbits, which are real-valued.
n = n_base   # Total number of nodes.

t = np.array(range(n))/n
x = np.array(range(n))+1

err = np.linalg.norm(x.imag)
print(err)

plt.plot(t,x)
plt.show()

# %%
# Therefore the FFT has Hermitian symmetry, meaning that the Fourier coefficients of negative index are the complex conjugates of their positive counterparts
fft_c = scipy.fft.fft(x)
err = (
    abs(fft_c[0] - fft_c[0].conjugate()) + 
    np.linalg.norm(fft_c[1:n//2] - fft_c[n//2+1:][::-1].conjugate())
)
print(err)

# %%
# This symmetry can be leveraged to only compute coefficients of positive index using the RFFT

rfft_c = scipy.fft.rfft(x)
err = np.linalg.norm(fft_c[:n//2+1] - rfft_c)
print(err)

# %%
# The positions are retrieved using the IRFFT.

irfft_c = scipy.fft.irfft(rfft_c)
err = np.linalg.norm(irfft_c - x)
print(err)

# %%
# Real-valued even signals and DCT I
# **********************************
# 
# Suppose a coordinate of the orbit is constrained to be even. 

n = 2*n_base   # Total number of nodes. Necessarily even here.
t = np.array(range(n))/n
y = np.array(range(n//2+1))+1
x = np.zeros(n)

for i in range(n//2+1):
    x[i] = y[i]

for i in range(n//2+1,n):
    x[i] = y[n - i]

z = np.zeros(n)
for i in range(n):
    z[i] = x[i] - x[(n-i) % n]

err = np.linalg.norm(z)
print(err)

plt.plot(t,x)
plt.show()
# %%
# In this case, the Fourier transform is purely real, i.e. its imaginary part is zero.
# 
rfft_c = scipy.fft.rfft(x)
err = np.linalg.norm(rfft_c.imag)
print(err)

# %%
# This symmetry can be leveraged to only compute the real part of the coefficients using the DCT I

dct_I_c = scipy.fft.dct(x[0:n//2+1],1)
err = np.linalg.norm(rfft_c.real - dct_I_c)
print(err)

# %%
# Half the positions are retrieved using the IDCT I.

idct_I_x = scipy.fft.idct(dct_I_c,1)
err = np.linalg.norm(x[0:n//2+1] - idct_I_x)
print(err)

# %%
# Real-valued odd signals and DST I
# *********************************
# 
# Suppose a coordinate of the orbit is constrained to be odd.

n = 2*n_base   # Total number of nodes. Necessarily even here.
t = np.array(range(n))/n
y = np.array(range(n//2-1))+1
x = np.zeros(n)

for i in range(1,n//2):
    x[i] = y[i-1]

for i in range(n//2+1,n):
    x[i] = - y[n - i - 1]

z = np.zeros(n)
for i in range(n):
    z[i] = x[i] + x[(n-i) % n]

err = np.linalg.norm(z)
print(err)

plt.plot(t,x)
plt.show()

# %%
# In this case, the Fourier transform is purely imaginary, i.e. its real part is zero.

rfft_c = scipy.fft.rfft(x)
err = np.linalg.norm(rfft_c.real)
print(err)

# %%
# This symmetry can be leveraged to only compute the imaginary part of the coefficients using the DST I

dst_I_c = scipy.fft.dst(x[1:n//2],1)
err = np.linalg.norm(rfft_c.imag[1:n//2] - (- dst_I_c))
print(err)

# %%
# Half the positions are retrieved using the IDST I.

idst_I_x = scipy.fft.idst(dst_I_c,1)
err = np.linalg.norm(x[1:n//2] - idst_I_x)
print(err)

# %%
# Real-valued even-odd signals and DCT III
# ****************************************
# 
# Suppose a coordinate of the orbit is constrained to be even arround the origin :math:`t=0` and odd arround a quarter period :math:`t=T/4`.

n = 4*n_base   # Total number of nodes. Necessarily divisible by 4 here.
t = np.array(range(n))/n
y = np.array(range(n//4))+1
x = np.zeros(n)
for i in range(n//4):
    x[i] = y[i]

for i in range(n//4+1,n//2):
    x[i] = - y[n//2 - i]

for i in range(n//2,n//2 + n//4):
    x[i] = - y[i-n//2]

for i in range(n//2 + n//4 + 1,n):
    x[i] = y[n - i]

z = np.zeros(n)
for i in range(n):
    z[i] = x[i] - x[(n-i) % n]

err = np.linalg.norm(z)
print(err)

for i in range(n):
    z[i] = x[i] + x[(n+n//2 - i) % n]

err = np.linalg.norm(z)
print(err)


plt.plot(t,x)
plt.show()

# %%
# In this case, the Fourier transform is purely real and its odd-numbered coefficients are zero.

rfft_c = scipy.fft.rfft(x)
err = np.linalg.norm(rfft_c.imag)
print(err)
err = np.linalg.norm(rfft_c[::2])
print(err)


# %%
# This symmetry can be leveraged to only compute the non-zero coefficients using the DCT III

dct_III_c = scipy.fft.dct(x[0:n//4],3)
err = np.linalg.norm(rfft_c.real[1::2] - 2*dct_III_c)
print(err)

# %%
# A quarter of the positions are retrieved using the IDCT III.

idct_III_c = scipy.fft.idct(dct_III_c,3)
err =  np.linalg.norm(x[0:n//4] - idct_III_c)
print(err)

# %%
# Real-valued odd-even signals and DST III
# ****************************************
#
# Suppose a coordinate of the orbit is constrained to be odd arround the origin :math:`t=0` and even arround a quarter period :math:`t=T/4`.

n = 4*n_base   # Total number of nodes. Necessarily divisible by 4 here.
t = np.array(range(n))/n
y = np.array(range(n//4))+1
x = np.zeros(n)

for i in range(1,n//4 + 1):
    x[i] = y[i-1]

for i in range(n//4+1,n//2):
    x[i] = y[n//2-1 - i]

for i in range(n//2+1,n//2 + n//4+1):
    x[i] = - y[i-(n//2+1)]

for i in range(n//2 + n//4+1,n):
    x[i] = -y[n-1 - i]

z = np.zeros(n)
for i in range(n):
    z[i] = x[i] + x[(n-i) % n]

err = np.linalg.norm(z)
print(err)

for i in range(n):
    z[i] = x[i] - x[(n+n//2 - i) % n]

err = np.linalg.norm(z)
print(err)

plt.plot(t,x)
plt.show()


# %%
# In this case, the Fourier transform is purely imaginary and its even-numbered coefficients are zero.

rfft_c = scipy.fft.rfft(x)
err = np.linalg.norm(rfft_c.real)
print(err)
err = np.linalg.norm(rfft_c[::2])
print(err)

# %%
# This symmetry can be leveraged to only compute the non-zero coefficients using the DST III

dst_III_c = scipy.fft.dst(x[1:n//4+1],3)
err = np.linalg.norm(rfft_c.imag[1::2] - (-2*dst_III_c))
print(err)

# %%
# A quarter of the positions are retrieved using the IDCT III.

idst_III_c = scipy.fft.idst(dst_III_c,3)
err = np.linalg.norm(x[1:n//4+1] - idst_III_c)
print(err)

# %%
# Real-valued signals with a subperiod
# ************************************
#
# Suppose a coordinate of the orbit is constrained to have a period that is a multiple of the base period

sub_period = 3
n = n_base * sub_period # Total number of nodes. Necessarily a multiple of the sub period.
t = np.array(range(n))/n
y = np.array(range(n_base))
x = np.zeros(n)

for i in range(sub_period):
    for j in range(n_base):
        x[j+i*n_base] = y[j]

z = np.zeros(n)
for i_sub in range(sub_period-1):
    for i in range(n):
        z[i] = x[i] - x[(i+(i_sub+1)*n_base) % n]

    err = np.linalg.norm(z)
    print(err)

plt.plot(t,x)
plt.show()

# %%
# In this case, the coefficients of the Fourier transform that are not a numbered by a multiple of the subperiod are zero.


rfft_c = scipy.fft.rfft(x)
err = 0
for i in range(1,sub_period):
    err += np.linalg.norm(rfft_c[i::sub_period])

print(err)

# %%
# This symmetry can be leveraged to only compute the non-zero coefficients using the RFFT on a sub-period only

rfftsub_c = scipy.fft.rfft(x[:n_base])
err = np.linalg.norm(rfft_c[::sub_period] - sub_period*rfftsub_c)
print(err)

# %%
# The positions on a sub-period are retrieved using the RFFT.

irfftsub_c = scipy.fft.irfft(rfftsub_c)
err = np.linalg.norm(x[:n_base] - irfftsub_c)
print(err)

