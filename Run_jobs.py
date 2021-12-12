import subprocess

# In order to kill all the created jobs : either quit the terminal or run the following command (possibly in another terminal):
# pkill -9 python



job = 'python Choreo_sniffall.py' 
# ~ job = 'python Choreo_target_custom.py' 

n=6
# ~ n=1

job_all = ''
for i in range(n):
    
    job_all = job_all + job
    
    # ~ if (i < n-1):
    if (True):
         job_all = job_all + ' & '

subprocess.run(job_all, shell=True)

