import subprocess
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


# In order to kill all the created jobs : either quit the terminal or run the following command (possibly in another terminal):
# pkill -9 python



# ~ job = 'python Choreo_sniffall.py' 
# ~ job = 'python Choreo_target_custom.py' 
job = 'python Choreo_target_custom2.py' 

n=10
# ~ n=1

job_all = ''
for i in range(n):
    
    job_all = job_all + job + ' & '

subprocess.run(job_all, shell=True)

