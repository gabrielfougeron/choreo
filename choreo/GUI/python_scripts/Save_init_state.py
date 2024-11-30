import os
import shutil
import numpy as np
import choreo 
import asyncio

import js
import pyodide

def NPY_JS_to_py(npy_js):

    return np.asarray(npy_js["data"]).reshape(npy_js["shape"])

async def main(params_dict):

    extra_args_dict = {}

    store_folder = '/Workspace/GUI solutions'

    if not(os.path.isdir(store_folder)):

        store_folder = 'Sniff_all_sym'

        if os.path.isdir(store_folder):
            shutil.rmtree(store_folder)
            os.makedirs(store_folder)
        else:
            os.makedirs(store_folder)

    file_basename = ''
    params_dict["Solver_Optim"]["n_opt"] = 0

    Workspace_folder = '/Workspace'
    extra_args_dict['store_folder'] = store_folder
    extra_args_dict['file_basename'] = file_basename
    extra_args_dict['save_first_init'] = True
    extra_args_dict['Save_SegmPos'] = True
    
    try:
        choreo.find.ChoreoChooseParallelEnvAndFind(Workspace_folder, params_dict, extra_args_dict)
    except Exception as exc:
        print("Error:", exc)
        
        js.postMessage(
            funname = "Error_From_Python",
            args    = pyodide.ffi.to_js(
                { },
                dict_converter=js.Object.fromEntries
            )
        )

        return

    filename_output = store_folder+'/_init'
    filename = filename_output+".json"
    
    if os.path.isfile(filename):

        with open(filename, 'rt') as fh:
            thefile = fh.read()
        os.remove(filename)
        
        blob = js.Blob.new([thefile], {type : 'application/text'})

        filename = filename_output+'.npy'
        all_pos = np.load(filename)
        os.remove(filename)

        js.postMessage(
            funname = "Play_Loop_From_Python",
            args    = pyodide.ffi.to_js(
                {
                    "is_sol":False,
                    "solname":"Non-solution initial state",
                    "JSON_data":blob,
                    "NPY_data":all_pos.reshape(-1),
                    "NPY_shape":all_pos.shape,
                    "DoClearScreen":True,
                    "DoXMinMax":True,
                    "ResetRot":True,
                },
                dict_converter=js.Object.fromEntries
            )
        )

    else:

        js.postMessage(
            funname = "Python_no_sol_found",
            args    = pyodide.ffi.to_js(
                {
                },
                dict_converter=js.Object.fromEntries
            )
        )
        
if __name__ == "__main__":
    
    params_dict = js.ConfigDict.to_py()
    asyncio.create_task(main(params_dict))


