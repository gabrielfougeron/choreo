import os
import numpy as np
import threadpoolctl
import choreo 
import json
import math
import shutil

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

def main():
        
    # Overwrite_Original = False
    Overwrite_Original = True
    
    # Remove_Different = False
    Remove_Different = True
    
    FindTimeReflexionIfPossible = True
    # FindTimeReflexionIfPossible = False
    
    refl_dim = [0,1]
    
    ntries = 10
    
    max_order = 2
    
    # skip_SymSig_if = lambda SymSig : False
    
    def skip_SymSig_if(SymSig):
        
        test_OK = True
        # test_OK = test_OK and (SymSig.BodyPerm[0] == 0)
        # # 
        # for i in range(3):
            # test_OK = test_OK and (SymSig.BodyPerm[i] == ((i+1)%3))
            
        test_OK = test_OK and (SymSig.BodyPerm[0] == 0)
        test_OK = test_OK and (SymSig.BodyPerm[1] == 2)
        test_OK = test_OK and (SymSig.BodyPerm[2] == 1)
        
        #     
        for i in range(2):
            test_OK = test_OK and (SymSig.SpaceRotSig[i] == -1)
        # 
        # test_OK = test_OK and (SymSig.TimeShiftNum  == 1 )
        test_OK = test_OK and (SymSig.TimeShiftDen  == 2)
        # 
        if test_OK:
            print(SymSig)
            print()
        
        return not test_OK
    
    hit_tol = 1e-9

    Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
    choreo.find.Load_wisdom_file(Wisdom_file)
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym")
    
    # input_folder = os.path.join(Workspace_folder, "test")    
    input_folder = os.path.join(Workspace_folder, "CLI solutions")    
    # input_folder = os.path.join(Workspace_folder, "ReflexionSymmetry")    
    
    # input_folder = os.path.join(Workspace_folder, "GUI solutions")    
    
    output_folder = os.path.join(Workspace_folder, "AdditionalSym")    
    params_filename = os.path.join(Workspace_folder, "choreo_config_reconverge.json")
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)
        
    params_dict["Solver_Optim"]["n_opt"] = 1
    # 
    all_files = os.listdir(input_folder)
    all_files.sort()
    
    # all_files = ["00009.json", "00040.json"]
    # all_files = ["00040.json"]
    all_files = ["00060.json"]
    
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
                
            FindTimeReflexion = (NBS.TimeRev > 0) and FindTimeReflexionIfPossible
            
            if FindTimeReflexion:
                res = choreo.find.FindTimeRevSymmetry(NBS, segmpos, ntries = ntries, refl_dim = refl_dim, hit_tol = hit_tol)
            else:
                res = choreo.find.FindTimeDirectSymmetry(NBS, segmpos, ntries = ntries, max_order = max_order, skip_SymSig_if = skip_SymSig_if, hit_tol = hit_tol)

            if res is None:
                print(f"Could not find {"reflexion" if FindTimeReflexion else "non-reflexion"} symmetry")
                No_SymFound_list.append(file_basename)
                continue
            
            else:
                print(f"Found {"reflexion" if FindTimeReflexion else "non-reflexion"} symmetry")
                print()
                Sym, segmpos = res
                
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
                "disp_scipy_opt" : True         ,
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
                    
            if Overwrite_Original and Solutions_are_same:

                for ext in ['.json','.npy','.png']:                        
                    try:
                        shutil.copyfile(full_out_file_basename+ext, full_in_file_basename+ext)
                        os.remove(full_out_file_basename+ext)
                        
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
