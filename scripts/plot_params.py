import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

# import choreo 

import numpy as np
import matplotlib.pyplot as plt

basename = 'Old'
# basename = 'New'

print(basename)
directory = os.path.join(__PROJECT_ROOT__, 'Sniff_all_sym', 'new_vs_old')
filename = os.path.join(directory, f'{basename}_params.npy')

params = np.load(filename)

x = np.log(np.abs(params))


print(x.shape)



fig, ax = plt.subplots()
ax.plot(x)
ax.set_yscale('log')
ax.set_ylim([1e-16, 1.])
plt.savefig(os.path.join(directory, f'{basename}_params.png'))