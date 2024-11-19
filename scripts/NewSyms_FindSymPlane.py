import os
import numpy as np
import choreo 
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

def main():
    
    DP_Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.txt")
    choreo.find.Load_wisdom_file(DP_Wisdom_file)
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym")
    
    input_folder = os.path.join(Workspace_folder, "CLI solutions")    
    # input_folder = os.path.join(Workspace_folder, "GUI solutions")    
    output_folder = os.path.join(Workspace_folder, "out")    
    
    params_filename = os.path.join(Workspace_folder, "choreo_config.json")
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)
        
    params_dict["Solver_Optim"]["n_opt"] = 1
    
    for thefile in os.listdir(input_folder):
        
        # if os.path.isfile(os.path.join(output_folder, thefile)):
        #     continue
            
        
        file_basename, ext = os.path.splitext(thefile)
        
        filename_output = os.path.join(output_folder, file_basename)
        
        if ext == '.json':
            
            print()
            print(file_basename)
            print()
            

            with open(os.path.join(input_folder,thefile)) as jsonFile:
                extra_args_dict = json.load(jsonFile)

            NBS, segmpos = choreo.NBodySyst.FromSolutionFile(os.path.join(input_folder, file_basename))
            params_buf = NBS.segmpos_to_params(segmpos)
            NBS.ForceGreaterNStore = True
            
            segmpos = NBS.params_to_segmpos(params_buf)
            
            Sym, segmpos_dt = choreo.find.FindReflectionSymmetry(NBS, segmpos)
            
            
            for ib in range(NBS.nbody):
                isegm = NBS.bodysegm[ib, 0]
                print(ib, isegm)
                
            
            # nint_fac_ini = NBS.nint_fac
            # params_buf = NBS.segmpos_to_params(segmpos_dt)
            # params_buf_fine = NBS.params_resize(params_buf, 2*nint_fac_ini)
            # NBS.nint_fac = 2*nint_fac_ini
            # segmpos_dt = NBS.params_to_segmpos(params_buf_fine)
            # NBS.nint_fac = nint_fac_ini
            
            Sym_list = [choreo.ActionSym.FromDict(Symp) for Symp in extra_args_dict["Sym_list"]]
            Sym_list.append(Sym)

            extra_args_dict.update({
                "store_folder"  : output_folder ,
                "nint_fac_init" : NBS.nint_fac //2 ,
                "ReconvergeSol" : True          ,
                "segmpos_ini"   : segmpos_dt    ,
                "Sym_list"      : Sym_list      ,
                "mass"          : np.array(extra_args_dict["bodymass"]),
                "charge"        : np.array(extra_args_dict["bodycharge"]),
                "AddNumberToOutputName" : False,
                "file_basename" : file_basename,
                "save_first_init" : True,
            })
            
            choreo.find.ChoreoFindFromDict(params_dict, extra_args_dict, Workspace_folder)

            break
            
            
            




if __name__ == "__main__":
    main()
