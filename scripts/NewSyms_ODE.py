import os
import numpy as np
import threadpoolctl
import choreo 
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))


# a = np.array([[0,1],[2,3]],dtype=np.float64)
# b = a.swapaxes(0, 1)
# 
# for i in range(2):
#     for j in range(2):
#         assert a[i,j] == b[j,i]
# 
# assert a.data.c_contiguous
# assert b.data.c_contiguous

def main():
    
    Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
    choreo.find.Load_wisdom_file(Wisdom_file)
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym")
    
    # input_folder = os.path.join(Workspace_folder, "CLI solutions")    
    input_folder = os.path.join(Workspace_folder, "GUI solutions")    
    output_folder = os.path.join(Workspace_folder, "out")    
    
    for thefile in os.listdir(input_folder):

        file_basename, ext = os.path.splitext(thefile)
        
        full_in_file_basename = os.path.join(input_folder, file_basename)
        full_out_file_basename = os.path.join(output_folder, file_basename)
        
        if ext == '.json':
            
            print()
            print(file_basename)
            print()

            with open(os.path.join(input_folder, thefile)) as jsonFile:
                extra_args_dict = json.load(jsonFile)

            NBS, segmpos = choreo.NBodySyst.FromSolutionFile(full_in_file_basename)
            params_buf = NBS.segmpos_to_params(segmpos)
            
            NBS.ForceGreaterNStore = True
            segmpos = NBS.params_to_segmpos(params_buf)
            
            ODE_Syst = NBS.Get_ODE_def(params_buf)
            
            nsteps = 10
            keep_freq = 10
            nint_ODE = (NBS.segm_store-1) * keep_freq
            method = "Gauss"
            
            rk = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method)
            
            segmpos_ODE, segmvel_ODE = choreo.scipy_plus.ODE.SymplecticIVP(
                rk = rk                 ,
                keep_freq = keep_freq   ,
                nint = nint_ODE         ,
                keep_init = True        ,
                **ODE_Syst              ,
            )
            
            segmpos_ODE = np.ascontiguousarray(segmpos_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))

            NBS.plot_segmpos_2D(segmpos_ODE, os.path.join(full_out_file_basename+'.png'))
            print(np.linalg.norm(segmpos - segmpos_ODE))


if __name__ == "__main__":
    with threadpoolctl.threadpool_limits(limits=1):
        main()
