import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

# 
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['TBB_NUM_THREADS'] = '1'


import choreo 


def main():
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym")
        
    choreo.find_new.ChoreoReadDictAndFind(Workspace_folder, config_filename="choreo_config.json")




if __name__ == "__main__":
    main()
