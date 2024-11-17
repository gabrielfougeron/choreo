import os
import numpy as np
import choreo 

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

def main():
    
    DP_Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.txt")
    choreo.find.Load_wisdom_file(DP_Wisdom_file)
    
    input_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym", "CLI solutions")    
    # input_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym", "GUI solutions")    
    output_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym", "out")    
    # choreo.find.ChoreoReadDictAndFind(Workspace_folder, config_filename="choreo_config.json")
    
    
    for thefile in os.listdir(input_folder):
        
        file_basename, ext = os.path.splitext(thefile)
        
        if ext == '.json':
            
            print()
            print(file_basename)
            print()
            
            NBS, segmpos = choreo.NBodySyst.FromSolutionFile(os.path.join(input_folder, file_basename))
            params_buf = NBS.segmpos_to_params(segmpos)
            action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)
            print(np.linalg.norm(action_grad))
            
            
            
            




if __name__ == "__main__":
    main()
