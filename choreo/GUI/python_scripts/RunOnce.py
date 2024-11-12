import os
import shutil
import asyncio
import numpy as np

import choreo 

import js
import pyodide

def Send_init_PlotInfo():

    file_basename = ''
    max_num_file = 0

    store_folder = '/Workspace/GUI solutions'

    if not(os.path.isdir(store_folder)):

        store_folder = 'Sniff_all_sym/'

    for filename in os.listdir(store_folder):

        file_path = os.path.join(store_folder, filename)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_basename in file_root) and (file_ext == '.json' ):

            file_root = file_root.replace(file_basename,"")

            try:
                max_num_file = max(max_num_file,int(file_root))
            except:
                pass

    max_num_file = max_num_file + 1
    n_find = max_num_file

    file_basename = file_basename+str(max_num_file).zfill(5)
    filename = os.path.join(store_folder,file_basename+'_init.json')

    if os.path.isfile(filename):

        with open(filename, 'rt') as fh:
            thefile = fh.read()

        blob = js.Blob.new([thefile], {type : 'application/text'})
    
        js.postMessage(
            funname = "Set_PlotInfo_From_Python",
            args    = pyodide.ffi.to_js(
                {
                    "JSON_data":blob,
                },
                dict_converter=js.Object.fromEntries
            )
        )
        
    else:
        
        raise ValueError('File not found')

def Plot_Loops_During_Optim_new(x, f, f_norm, NBS, jacobian):

    AABB = NBS.GetFullAABB(jacobian.segmpos, 0., MakeSquare=True)

    windowObject = {
        "xMin": AABB[0,0]   ,
        "xMax": AABB[1,0]   ,   
        "yMin": AABB[0,1]   ,
        "yMax": AABB[1,1]   ,
    }

    js.postMessage(

        funname = "Plot_Loops_During_Optim_From_Python",
        args    = pyodide.ffi.to_js(
            {
                "NPY_data":jacobian.segmpos.reshape(-1),
                "NPY_shape":jacobian.segmpos.shape,
                "Current_PlotWindow":windowObject
            },
            dict_converter=js.Object.fromEntries
        )
    )
    
def Plot_Loops_During_Optim(x,f,f_norm,ActionSyst):
    
    xmin,xmax,ymin,ymax = ActionSyst.HeuristicMinMax()

    hside = max(xmax-xmin,ymax-ymin)/2

    xmid = (xmin+xmax)/2
    ymid = (ymin+ymax)/2

    windowObject = {}

    windowObject["xMin"] = xmid - hside
    windowObject["xMax"] = xmid + hside

    windowObject["yMin"] = ymid - hside
    windowObject["yMax"] = ymid + hside

    
    js.postMessage(

        funname = "Plot_Loops_During_Optim_From_Python",
        args    = pyodide.ffi.to_js(
            {
                "NPY_data":ActionSyst.last_all_pos.reshape(-1),
                "NPY_shape":ActionSyst.last_all_pos.shape,
                "Current_PlotWindow":windowObject
            },
            dict_converter=js.Object.fromEntries
        )
    )

def ListenToNextFromGUI(x,f,f_norm,ActionSyst):

    AskForNext =  (js.AskForNext.to_py()[0] == 1)

    js.AskForNext[0] = 0

    return AskForNext

def ListenToNextFromGUI_new(x,f,f_norm,ActionSyst,jacobian):

    AskForNext =  (js.AskForNext.to_py()[0] == 1)

    js.AskForNext[0] = 0

    return AskForNext

def NPY_JS_to_py(npy_js):

    return np.asarray(npy_js["data"]).reshape(npy_js["shape"])

async def main(params_dict):
    
    extra_args_dict = {}

    callback_after_init_list = []

    if params_dict['Animation_Search']['DisplayLoopsDuringSearch']:
        callback_after_init_list.append(Send_init_PlotInfo)

    optim_callback_list = [ListenToNextFromGUI_new]

    if params_dict['Animation_Search']['DisplayLoopsDuringSearch']:

        optim_callback_list.append(Plot_Loops_During_Optim_new)

    extra_args_dict['callback_after_init_list'] = callback_after_init_list
    extra_args_dict['optim_callback_list'] = optim_callback_list

    store_folder = '/Workspace/GUI solutions'

    if not(os.path.isdir(store_folder)):

        store_folder = 'Sniff_all_sym/'

        if os.path.isdir(store_folder):
            shutil.rmtree(store_folder)
            os.makedirs(store_folder)
        else:
            os.makedirs(store_folder)

    file_basename = ''

    max_num_file = 0
    
    for filename in os.listdir(store_folder):
        file_path = os.path.join(store_folder, filename)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_basename in file_root) and (file_ext == '.json' ):

            file_root = file_root.replace(file_basename,"")

            try:
                max_num_file = max(max_num_file,int(file_root))
            except:
                pass

    max_num_file = max_num_file + 1

    file_basename = file_basename+str(max_num_file).zfill(5)

    Workspace_folder = '/Workspace'
    extra_args_dict['store_folder'] = store_folder
    extra_args_dict['max_num_file'] = max_num_file
    extra_args_dict['file_basename'] = file_basename
    
    try:
        choreo.find.ChoreoChooseParallelEnvAndFind(Workspace_folder, params_dict, extra_args_dict)
    except Exception as exc:
        print(exc)

    filename_output = store_folder+'/'+file_basename
    filename = filename_output+".json"
    
    if os.path.isfile(filename):

        with open(filename, 'rt') as fh:
            thefile = fh.read()

        blob = js.Blob.new([thefile], {type : 'application/text'})

        filename = filename_output+'.npy'
        all_pos = np.load(filename)

        js.postMessage(
            funname = "Play_Loop_From_Python",
            args    = pyodide.ffi.to_js(
                {
                    "is_sol":True,
                    "solname":"User generated solution: "+file_basename,
                    "JSON_data":blob,
                    "NPY_data":all_pos.reshape(-1),
                    "NPY_shape":all_pos.shape,
                    "DoClearScreen":not(params_dict['Animation_Search']['DisplayLoopsDuringSearch']),
                    "DoXMinMax":not(params_dict['Animation_Search']['DisplayLoopsDuringSearch']),
                    "ResetRot":False,
                },
                dict_converter=js.Object.fromEntries
            )
        )

    else:

        print("Solver did not find a solution.")

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


