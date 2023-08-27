"""
Description of the benchmark machine
====================================
"""

# %%
# Description of the machine on which benchmarks were run.

import subprocess
 
# traverse the info
Id = subprocess.check_output(['lshw']).decode('utf-8').split('\n')

for line in Id:
    print(line)
    

# %%
# Numpy config informations

import numpy as np
np.show_config()