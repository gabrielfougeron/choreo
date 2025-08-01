import os
import numpy as np
import threadpoolctl
import choreo 
import json
import math

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

def main():
        
    Remove_Original = False
    # Remove_Original = True
    
    # Remove_Different = False
    Remove_Different = True
    
    max_order = 11
    
    # skip_SymSig_if = lambda SymSig : False
    
    def skip_SymSig_if(SymSig):
        return SymSig.TimeShiftDen != 11
    
    
    Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
    choreo.find.Load_wisdom_file(Wisdom_file)
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym")
    
    # input_folder = os.path.join(Workspace_folder, "test")    
    input_folder = os.path.join(Workspace_folder, "CLI solutions")    
    # input_folder = os.path.join(Workspace_folder, "GUI solutions")    
    
    
    # output_folder = os.path.join(Workspace_folder, "ReflexionSymmetry")    
    output_folder = os.path.join(Workspace_folder, "TimeDirectSymmetry")    
    params_filename = os.path.join(Workspace_folder, "choreo_config_reconverge.json")
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)
        
    params_dict["Solver_Optim"]["n_opt"] = 1
    
    all_files = os.listdir(input_folder)
    all_files.sort()
    
    different_list = []
    No_cvgence_list = []
    No_SymFound_list = []
    
    for thefile in all_files:
        
        file_basename, ext = os.path.splitext(thefile)
        if ext != '.json':
            continue
        
        full_in_file_basename = os.path.join(input_folder, file_basename)
        full_out_file_basename = os.path.join(output_folder, file_basename)
                    
        print()
        print(file_basename)

        NBS, segmpos = choreo.NBodySyst.FromSolutionFile(full_in_file_basename)
        
        if not os.path.isfile(os.path.join(output_folder, thefile)):

            with open(os.path.join(input_folder, thefile)) as jsonFile:
                extra_args_dict = json.load(jsonFile)

            params_buf = NBS.segmpos_to_params(segmpos)
            
            # nint_new = math.lcm(NBS.nint, *list(range(1,max_order+1)))
            nint_new = math.lcm(NBS.nint, max_order)
            nint_fac_new = nint_new // (2 * NBS.nint_min)

            params_buf = NBS.params_resize(params_buf, nint_fac_new)
            NBS.nint_fac = nint_fac_new

            NBS.ForceGreaterNStore = True            
            segmpos = NBS.params_to_segmpos(params_buf)
            
            Sym = choreo.find.FindTimeDirectSymmetry(NBS, segmpos, ntries = 10, max_order = max_order, skip_SymSig_if = skip_SymSig_if)
            
            if Sym is None:
                print("Could not find TimeDirect symmetry")
                No_SymFound_list.append(file_basename)
                continue
            
            Sym_list = [choreo.ActionSym.FromDict(Symp) for Symp in extra_args_dict["Sym_list"]]
            Sym_list.append(Sym)

            extra_args_dict.update({
                "store_folder"  : output_folder ,
                "ReconvergeSol" : True          ,
                "nint_fac_init" : None          ,
                "NBS_ini"       : NBS           ,
                "segmpos_ini"   : segmpos       ,
                "Sym_list"      : Sym_list      ,
                "mass"          : np.array(extra_args_dict["bodymass"]),
                "charge"        : np.array(extra_args_dict["bodycharge"]),
                "AddNumberToOutputName" : False ,
                "file_basename" : file_basename ,
                # "save_first_init" : True        ,
                "Look_for_duplicates" : False   ,
            })
        
            choreo.find.ChoreoFindFromDict(params_dict, extra_args_dict, Workspace_folder)
            
        if os.path.isfile(full_out_file_basename + '.json'): # Solution was found
        
            with open(os.path.join(input_folder, thefile)) as jsonFile:
                in_sol_dict = json.load(jsonFile)
                
            with open(full_out_file_basename + '.json') as jsonFile:
                out_sol_dict = json.load(jsonFile)
            
            print(f"Solution grad norm : {out_sol_dict['Grad_Action']}")
            
            Hash_in = np.array(in_sol_dict["Hash"])
            Hash_out = np.array(out_sol_dict["Hash"])
            
            Solutions_are_same = NBS.TestHashSame(Hash_in, Hash_out)
            
            if not Solutions_are_same:
                print("WARNING : Found solution is not the same")
                
                different_list.append(file_basename)
                
                if Remove_Different:
                        
                    for ext in ['.json','.npy','.png']:                        
                        try:
                            os.remove(full_out_file_basename+ext)
                        except:
                            pass
                    
            if Remove_Original and Solutions_are_same:

                for ext in ['.json','.npy','.png']:                        
                    try:
                        os.remove(full_in_file_basename+ext)
                    except:
                        pass
                    
        else:
            No_cvgence_list.append(file_basename)

    if len(No_SymFound_list) > 0:
        print()
        print("WARNING : Could not find additional symmetry for the following solutions :")
        for name in No_SymFound_list:
            print(name)
            
    if len(different_list) > 0:
        print()
        print("WARNING : The following initial solutions reconverged to different symmetrized solutions :")
        for name in different_list:
            print(name)

    if len(No_cvgence_list) > 0:
        print()
        print("WARNING : The following initial solutions could not reconverge :")
        for name in No_cvgence_list:
            print(name)

if __name__ == "__main__":
    with threadpoolctl.threadpool_limits(limits=1):
        main()
