import os
import subprocess
import time

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

examples_dir = os.path.join(__PROJECT_ROOT__,"examples")

env = os.environ
env['DISPLAY'] = ''

env['OPENBLAS_NUM_THREADS'] = '1'
env['NUMEXPR_NUM_THREADS'] = '1'
env['MKL_NUM_THREADS'] = '1'
env['OMP_NUM_THREADS'] = '1'

for root, dirs, files in os.walk(examples_dir):
    for name in files:
        full_name = os.path.join(root, name)
        base, ext = os.path.splitext(full_name)

        if ext in [".py"]:
            print()
            print(f"Running {name}")
            
            tbeg = time.perf_counter()
            subprocess.run(
                ["python", full_name],
                # capture_output=True,
                env = env,
            )
            tend = time.perf_counter()
            
            print(f'Ran in {tend-tbeg}')
