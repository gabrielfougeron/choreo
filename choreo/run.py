"""
Toto



"""


import os
import sys
import multiprocessing
import choreo

import argparse

def CLI_search(cli_args):

    parser = argparse.ArgumentParser(
        description = 'Searches periodic solutions to the N-body problem as defined in choreo_GUI')

    default_Workspace = './'
    parser.add_argument(
        '-f', '--foldername',
        default = default_Workspace,
        dest = 'Workspace_folder',
        help = f'Workspace folder as defined in the GUI. Defaults to the current directory.',
        metavar = '',
    )

    args = parser.parse_args(cli_args)

    root_list = [
        '',
        os.getcwd(),
    ]

    FoundFile = False
    for root in root_list:

        Workspace_folder = os.path.join(root, args.Workspace_folder)

        if os.path.isdir(Workspace_folder):
            FoundFile = True
            break

    Wisdom_file = os.path.join(Workspace_folder, "PYFFTW_wisdom.json")
    choreo.find.Load_wisdom_file(Wisdom_file)

    if (FoundFile):

        os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        choreo.find.ChoreoReadDictAndFind(Workspace_folder)

    else:

        print(f'Workspace folder {args.Workspace_folder} was not found')

def entrypoint_CLI_search():
    """ Search in CLI 
    
    Toto
    
    """
    CLI_search(sys.argv[1:])
