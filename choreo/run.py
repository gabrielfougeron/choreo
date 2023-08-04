import os
import sys
import multiprocessing
import choreo

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

import argparse

def GUI_in_CLI(cli_args):

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

        sys.path.append(__PROJECT_ROOT__)

        choreo.find.ChoreoReadDictAndFind(Workspace_folder)

    else:

        print(f'Workspace folder {args.Workspace_folder} was not found')

def GUI_in_CLI_old(cli_args):

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

        sys.path.append(__PROJECT_ROOT__)

        choreo.find.ChoreoReadDictAndFind_old(Workspace_folder)

    else:

        print(f'Workspace folder {args.Workspace_folder} was not found')

def entrypoint_GUI_in_CLI():
    GUI_in_CLI(sys.argv[1:])

if __name__ == '__main__':
    entrypoint_GUI_in_CLI()