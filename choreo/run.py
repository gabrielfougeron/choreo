import os
import sys
import multiprocessing
import choreo

from choreo.GUI import default_gallery_root, install_official_gallery

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

        Workspace_folder = os.path.join(root,args.Workspace_folder)

        if os.path.isdir(Workspace_folder):
            FoundFile = True
            break

    if (FoundFile):

        os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        choreo.find.ChoreoReadDictAndFind(Workspace_folder)

    else:

        print(f'Workspace folder {args.Workspace_folder} was not found')

def entrypoint_CLI_search():
    CLI_search(sys.argv[1:])

def GUI(cli_args):
    
    parser = argparse.ArgumentParser(
        description = 'Launches choreo GUI')

    default_Workspace = './'
    parser.add_argument(
        '-f', '--foldername',
        default = default_gallery_root,
        dest = 'gallery_root',
        help = f'Gallery root.',
        metavar = '',
    )

    args = parser.parse_args(cli_args)
    
    if args.gallery_root != default_gallery_root:
        raise NotImplementedError
    
    GalleryExists = (
        os.path.isdir(os.path.join(args.gallery_root,'choreo-gallery'))
        and os.path.isfile(os.path.join(args.gallery_root,'gallery_descriptor.json'))
    )
    
    if not GalleryExists:
        print("Gallery not found. Installing official gallery.")
        install_official_gallery()
    
    dist_dir = os.path.join(args.gallery_root,'python_dist')
    
    if os.path.isdir(dist_dir):
        
        for f in os.listdir(dist_dir):
            if ('.whl' in f) and ('pyodide' in f):
                FoundPyodideWheel = True
                break
        else:
            FoundPyodideWheel = False
    else:
        FoundPyodideWheel = False
    
    if not FoundPyodideWheel:
        print("Warning : Pyodide wheel not found")
    
    choreo.GUI.serve_GUI()


def entrypoint_GUI():
    GUI(sys.argv[1:])
