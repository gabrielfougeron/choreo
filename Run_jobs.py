import subprocess

# ~ job = 'python Choreo_sniffall.py' 
job = 'python Choreo_target_custom.py' 

n=6

job_all = ''
for i in range(n):
    
    job_all = job_all + job
    
    # ~ if (i < n-1):
    if (True):
         job_all = job_all + ' & '

subprocess.run(job_all, shell=True)

