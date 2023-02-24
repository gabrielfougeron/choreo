import os


import numpy as np
import scipy
import scipy.fft
import perfplot
import functools


def setup_top(n,nd):

    dim_list = [n for i in range(nd)]
    dim_list.append(2)

    all_coeffs_d = np.random.rand(*dim_list)
    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]

    return c_coeffs_d



fft_fun = scipy.fft.irfft
# fft_fun = scipy.fft.ihfft
# fft_fun = scipy.fft.rfft


nd = 3

setup = functools.partial(setup_top,nd=nd)

kernels = [ functools.partial(fft_fun,axis=axis) for axis in range(nd) ]

labels = [ str(axis) for axis in range(nd) ]


out = perfplot.bench(
    setup=setup,
    kernels=kernels,
    labels=labels,
    n_range=[2**k for k in range(3,8)],
    # n_range=[2**k for k in range(25)],
    xlabel="len(a)",
    # More optional arguments with their default values:
    # logx="auto",  # set to True or False to force scaling
    # logy="auto",
    equality_check=None,  # set to None to disable "correctness" assertion
    # show_progress=True,
    target_time_per_measurement=10.0,
    # max_time=None,  # maximum time per measurement
    # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
    # flops=lambda n: 3*n,  # FLOPS plots
)

out.save(
    filename = "perf.png",
    transparent = False,
    relative_to=0,  # plot the timings relative to one of the measurements
    )