import os
import numpy as np
import threadpoolctl
import choreo 
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

# TODO : Add this as pytests

def main():
    
    eps = 1e-12
        
    Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
    choreo.find.Load_wisdom_file(Wisdom_file)
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym")
    
    input_folder = os.path.join(Workspace_folder, "CLI solutions")    
    # input_folder = os.path.join(Workspace_folder, "GUI solutions")    
    output_folder = os.path.join(Workspace_folder, "out")    
    
    params_filename = os.path.join(Workspace_folder, "choreo_config.json")
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)
        
    params_dict["Solver_Optim"]["n_opt"] = 1
    
    for thefile in os.listdir(input_folder):
        
        if os.path.isfile(os.path.join(output_folder, thefile)):
            continue
            
        file_basename, ext = os.path.splitext(thefile)
        
        full_in_file_basename = os.path.join(input_folder, file_basename)

        nfac = 30
        
        if ext == '.json':
            
            print()
            print(file_basename)
            print()

            with open(os.path.join(input_folder, thefile)) as jsonFile:
                extra_args_dict = json.load(jsonFile)

            NBS, segmpos = choreo.NBodySyst.FromSolutionFile(full_in_file_basename)
            params_buf = NBS.segmpos_to_params(segmpos)

            des_old = NBS.Segmpos_Descriptor(params_buf)
            hash_old = np.array(des_old["Hash"])
            action_old = des_old["Action"]
            minbindist_old = np.array(des_old["Min_Bin_Distance"])
            looplength_old = np.array(des_old["Loop_Length"])
            
                
                
            # Resize
                
                
            nint_fac_new = nfac*NBS.nint_fac
            params_buf_new = NBS.params_resize(params_buf, nint_fac_new)
            NBS.nint_fac = nint_fac_new

            des_new = NBS.Segmpos_Descriptor(params_buf_new)
            hash_new = np.array(des_new["Hash"])
            action_new = des_new["Action"]
            minbindist_new = np.array(des_new["Min_Bin_Distance"])
            looplength_new = np.array(des_new["Loop_Length"])
            
            
            
            # Comparison
            
            print(np.linalg.norm(hash_old - hash_new))
            assert np.linalg.norm(hash_old - hash_new) < eps
            
            print(abs(action_old - action_new))
            assert abs(action_old - action_new) < eps            
            
            print(np.linalg.norm(minbindist_old - minbindist_new))
            # assert np.linalg.norm(minbindist_old - minbindist_new) < eps
            
            print(np.linalg.norm(looplength_old - looplength_new))
            assert np.linalg.norm(looplength_old - looplength_new) < eps
            
            
            
            




if __name__ == "__main__":
    with threadpoolctl.threadpool_limits(limits=1):
        main()
