import os
import multiprocessing

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

import argparse

parser = argparse.ArgumentParser(
    description = 'Searches periodic solutions to the N-body problem as defined in choreo_GUI')

parser.add_argument(
    '-f', '--filename',
    default = 'Sniff_all_sym/',
    dest = 'Workspace_folder',
    help = 'Workspace folder'
)

args = parser.parse_args()

root_list = [
    '',
    __PROJECT_ROOT__,
    os.getcwd(),
]

FoundFile = False
for root in root_list:

    Workspace_folder = os.path.join(root,args.Workspace_folder)

    if os.path.isdir(Workspace_folder):
        FoundFile = True
        break

if (FoundFile):

    os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    import sys

    sys.path.append(__PROJECT_ROOT__)

    import choreo 
    choreo.ChoreReadDictAndFind(Workspace_folder)

else:

    print(f'Workspace folder {args.Workspace_folder} was not found')