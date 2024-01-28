import pyfftw
import numpy as np
import multiprocessing
import itertools
import os
import tqdm

DP_Wisdom_file = "PYFFTW_wisdom.txt"

Load_wisdom = True
# Load_wisdom = False
# 
# Write_wisdom = True
Write_wisdom = False



if Load_wisdom:
    if os.path.isfile(DP_Wisdom_file):
        with open(DP_Wisdom_file, 'rb') as f:
            wis_list = f.readlines()
            
    wis = tuple(wis_list)
    pyfftw.import_wisdom(wis)

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(300000)


def setup_all(fft_type, nthreads, n):
    
    simd_n = pyfftw.simd_alignment
    

    planner_effort = 'FFTW_WISDOM_ONLY'
    # planner_effort = 'FFTW_ESTIMATE'
    # planner_effort = 'FFTW_MEASURE'   
    # planner_effort = 'FFTW_PATIENT'
    # planner_effort = 'FFTW_EXHAUSTIVE'

    pyfftw.config.NUM_THREADS = nthreads
    pyfftw.config.PLANNER_EFFORT = planner_effort

    if fft_type in ['fft']:
        x = pyfftw.empty_aligned((n), dtype='complex128', n=simd_n)
        y = pyfftw.empty_aligned((n), dtype='complex128', n=simd_n)
        direction = 'FFTW_FORWARD'
    elif fft_type in ['rfft']:
        x = pyfftw.empty_aligned((n), dtype='float64', n=simd_n)
        m = n//2 + 1
        y = pyfftw.empty_aligned((m), dtype='complex128', n=simd_n)
        direction = 'FFTW_FORWARD'
    else:
        raise ValueError(f'No prepare function for {fft_type}')
    
    fft_object = pyfftw.FFTW(x, y, axes=(0, ), direction=direction, flags=(planner_effort,), threads=nthreads, planning_timelimit=None)

    return fft_object
    
    
all_fft_types = [
    'fft',
    'rfft',
]

all_nthreads = [
    1, 
    multiprocessing.cpu_count()//2
]

all_sizes = np.array([4*3*5 * 2**n for n in range(15)])
    
total_it = len(all_fft_types) * len(all_nthreads) * len(all_sizes)
    
    
all_custom = []    
with tqdm.tqdm(total=total_it) as progress_bar:
    for fft_type, nthreads, n in itertools.product(all_fft_types, all_nthreads, all_sizes):
        
        # print(f'{fft_type = } {nthreads = } {n = }')
        
        all_custom.append(setup_all(fft_type, nthreads, n))
        
        progress_bar.update(1)
    
    
if Write_wisdom:    
    wis = pyfftw.export_wisdom()

    with open(DP_Wisdom_file, 'wb') as f:
        for i in range(3):
            f.write(wis[i])
    