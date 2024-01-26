import pyfftw
import numpy as np
import multiprocessing
import itertools
import os
import tqdm

DP_Wisdom_file = "PYFFTW_wisdom.txt"

# Load_wisdom = True
Load_wisdom = False

if Load_wisdom:
    if os.path.isfile(DP_Wisdom_file):
        with open(DP_Wisdom_file, 'rb') as f:
            wis_dp = f.read()
            
        print(wis_dp)

    wis_old = pyfftw.export_wisdom()
    wis_new = (wis_dp, wis_old[1], wis_old[2])
    pyfftw.import_wisdom(wis_new)


pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(300000)


def setup_all(fft_type, nthreads, n):

    # planner_effort = 'FFTW_ESTIMATE'
    # planner_effort = 'FFTW_MEASURE'   
    planner_effort = 'FFTW_PATIENT'
    # planner_effort = 'FFTW_EXHAUSTIVE'

    pyfftw.config.NUM_THREADS = nthreads
    pyfftw.config.PLANNER_EFFORT = planner_effort

    if fft_type in ['fft']:
        x = np.random.random(n) + 1j*np.random.random(n)
        y = np.random.random(n) + 1j*np.random.random(n)
        direction = 'FFTW_FORWARD'
    elif fft_type in ['rfft']:
        x = np.random.random(n)
        m = n//2 + 1
        y = np.random.random(m) + 1j*np.random.random(m)
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
    
    
    
wis = pyfftw.export_wisdom()

wis_dp = wis[0]
wis_sp = wis[1]
wis_ldp = wis[2]

with open(DP_Wisdom_file, 'wb') as f:
    f.write(wis_dp)
    