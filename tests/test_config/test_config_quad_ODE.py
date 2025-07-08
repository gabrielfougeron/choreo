import numpy as np
import scipy
import choreo
from tests.test_config.test_config_NBodySyst import load_from_solution_file

Small_orders = list(range(2,11))
High_orders = [25,50,100]

all_fun_types = [
    "py_fun"            ,
    "c_fun_memoryview"  ,
    "c_fun_pointer"     ,
]
     
StableInterpMethods = [
    "Gauss"             ,
    "Radau_I"           ,
    "Radau_II"          ,
    "Lobatto_III"       ,
    "Cheb_I"            ,
    "Cheb_II"           ,
    "ClenshawCurtis"    ,
]  
       
QuadMethods = [
    "Gauss"             ,
    "Radau_I"           ,
    "Radau_II"          ,
    "Lobatto_III"       ,
    "Cheb_I"            ,
    "Cheb_II"           ,
    "ClenshawCurtis"    ,
    "NewtonCotesOpen"   ,
    "NewtonCotesClosed" ,
]    

all_quad_problem_names = [
    "Wallis"    ,
]

def define_quad_problem(eq_name):
        
    if eq_name == "Wallis":

        def nint(th_cvg_rate):
            
            if th_cvg_rate > 20:
                return 1
            elif th_cvg_rate > 7:
                return 10
            elif th_cvg_rate > 4:
                return 100
            elif th_cvg_rate > 2:
                return 2000
            else:
                return 100000
            
        x_span = (0., np.pi/2)
        
        ndim = 10

        ex_sol = np.empty((ndim), dtype=np.float64)
        for i in range(ndim):
            ex_sol[i] = scipy.special.beta((i+1)/2, 1/2)/2

        def py_fun(x):
            res = np.empty((ndim), dtype=np.float64)
            res[0] = 1.
            s = np.sin(x)
            for i in range(1,ndim):
                res[i] = res[i-1] * s
            return res
            
        def py_fun_inplace(x,res):
            res[0] = 1.
            s = np.sin(x)
            for i in range(1,ndim):
                res[i] = res[i-1] * s
                 
        c_fun_pointer = choreo.segm.quad.nb_jit_inplace_double_array(py_fun_inplace)
        c_fun_memoryview = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "Wallis_c_fun_memoryview")
        
        return {
            "ndim" : ndim               ,
            "nint" : nint               ,
            "x_span" : x_span           ,
            "ex_sol" : ex_sol           ,
            "fun" : {
                "py_fun" : py_fun                       ,                                       
                "c_fun_pointer" : c_fun_pointer         ,
                "c_fun_memoryview" : c_fun_memoryview   ,
            }                           ,
        }
        
    else:
        raise ValueError(f'Unknown {eq_name = }')

ClassicalImplicitRKMethods = [
    "Gauss"             ,
    "Radau_IA"          ,
    "Radau_IIA"         ,
    "Radau_IB"          ,
    "Radau_IIB"         ,
    "Lobatto_IIIA"      ,
    "Lobatto_IIIB"      ,
    "Lobatto_IIIC"      ,
    "Lobatto_IIIC*"     ,
    "Lobatto_IIID"      ,            
    "Lobatto_IIIS"      ,     
    # "Cheb_I"            ,  # Fishy things happening for those. TODO : Investigate.
    # "Cheb_II"           ,
    # "ClenshawCurtis"    ,
    # "NewtonCotesOpen"   ,
    # "NewtonCotesClosed" ,
]

SymplecticImplicitRKMethodPairs = [
    ("Gauss"        , "Gauss"           ),
    ("Radau_IB"     , "Radau_IB"        ),
    ("Radau_IIB"    , "Radau_IIB"       ),
    ("Lobatto_IIIA" , "Lobatto_IIIB"    ),
    ("Lobatto_IIIC" , "Lobatto_IIIC*"   ),
    ("Lobatto_IIID" , "Lobatto_IIID"    ),
    ("Lobatto_IIIS" , "Lobatto_IIIS"    ),
]   

SymmetricImplicitRKMethodPairs = [
    ("Gauss"        , "Gauss"           ),
    ("Lobatto_IIIA" , "Lobatto_IIIA"    ),
    ("Radau_IB"     , "Radau_IIB"       ),
    ("Lobatto_IIIS" , "Lobatto_IIIS"    ),
    ("Lobatto_IIID" , "Lobatto_IIID"    ),
]  

SymmetricSymplecticImplicitRKMethodPairs = [
    ("Radau_IA"     , "Radau_IIA"       ),
]   

Explicit_tables_dict = {
    "SymplecticEuler"   : choreo.segm.precomputed_tables.SymplecticEuler,
    "StormerVerlet"     : choreo.segm.precomputed_tables.StormerVerlet  ,
    "McAte2"            : choreo.segm.precomputed_tables.McAte2         ,
    "Ruth3"             : choreo.segm.precomputed_tables.Ruth3          ,
    "McAte3"            : choreo.segm.precomputed_tables.McAte3         ,
    "Ruth4"             : choreo.segm.precomputed_tables.Ruth4          ,
    "Ruth4Rat"          : choreo.segm.precomputed_tables.Ruth4Rat       ,
    "McAte4"            : choreo.segm.precomputed_tables.McAte4         ,
    "CalvoSanz4"        : choreo.segm.precomputed_tables.CalvoSanz4     ,
    "McAte5"            : choreo.segm.precomputed_tables.McAte5         ,
    "Yoshida6A"         : choreo.segm.precomputed_tables.Yoshida6A      ,
    "Yoshida6B"         : choreo.segm.precomputed_tables.Yoshida6B      ,
    "Yoshida6C"         : choreo.segm.precomputed_tables.Yoshida6C      ,
    "KahanLi6"          : choreo.segm.precomputed_tables.KahanLi6       ,
    "McLahan8"          : choreo.segm.precomputed_tables.McLahan8       ,
    "KahanLi8"          : choreo.segm.precomputed_tables.KahanLi8       ,
    "Yoshida8A"         : choreo.segm.precomputed_tables.Yoshida8A      ,
    "Yoshida8B"         : choreo.segm.precomputed_tables.Yoshida8B      ,
    "Yoshida8C"         : choreo.segm.precomputed_tables.Yoshida8C      ,
    "Yoshida8D"         : choreo.segm.precomputed_tables.Yoshida8D      ,
    "Yoshida8E"         : choreo.segm.precomputed_tables.Yoshida8E      ,
    "SofSpa10"          : choreo.segm.precomputed_tables.SofSpa10       ,
}

all_ODE_names = [
    "ypp=minus_y"       ,
    "ypp=xy"            ,
]

def define_ODE_ivp(eq_name):
        
    if eq_name == "ypp=minus_y":
        # y'' = - y
        # y(x) = A cos(x) + B sin(x)

        def nint(th_cvg_rate):

            if th_cvg_rate > 20:
                return 1
            elif th_cvg_rate > 10:
                return 2
            elif th_cvg_rate > 7:
                return 25
            elif th_cvg_rate > 5:
                return 50
            elif th_cvg_rate > 4:
                return 200
            
        t_span = (0., 1.)
        
        ndim = 2

        ex_sol_fun_x = lambda t : np.array([ np.cos(t), np.sin(t)])
        ex_sol_fun_v = lambda t : np.array([-np.sin(t), np.cos(t)])

        def py_fun(t,v):
            return np.asarray(v)
        
        def py_gun(t,x):
            return -np.asarray(x)
        
        py_fun_vec = py_fun
        py_gun_vec = py_gun
        
        def py_fun_inplace(t,v,res):
            for i in range(ndim):
                res[i] = v[i]   
                 
        def py_gun_inplace(t,x,res):
            for i in range(ndim):
                res[i] = -x[i]   
        
        c_fun_pointer = choreo.segm.ODE.nb_jit_c_fun_pointer(py_fun_inplace)
        c_gun_pointer = choreo.segm.ODE.nb_jit_c_fun_pointer(py_gun_inplace)
        
        c_fun_memoryview = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_my_c_fun_memoryview")
        c_gun_memoryview = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_my_c_gun_memoryview")
        
        c_fun_memoryview_vec = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_my_c_fun_memoryview_vec")
        c_gun_memoryview_vec = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_my_c_gun_memoryview_vec")        
        
    elif eq_name == "ypp=xy":

        def nint(th_cvg_rate):

            if th_cvg_rate > 20:
                return 1
            elif th_cvg_rate > 10:
                return 2
            elif th_cvg_rate > 7:
                return 30
            elif th_cvg_rate > 5:
                return 50
            elif th_cvg_rate > 4:
                return 200
            
        t_span = (0., 1.)
        
        ndim = 2
        
        def ex_sol_fun_x(t):
            ai, _, bi, _ = scipy.special.airy(t)
            return np.array([ai,bi])    
            
        def ex_sol_fun_v(t):
            _, aip, _, bip = scipy.special.airy(t)
            return np.array([aip,bip])

        def py_fun(t,v):
            return np.asarray(v)
        
        def py_gun(t,x):
            return t*np.asarray(x)
        
        py_fun_vec = py_fun
        
        def py_gun_vec(t,x):
            return np.einsum('i,ij->ij', np.asarray(t), np.asarray(x))
        
        def py_fun_inplace(t,v,res):
            for i in range(ndim):
                res[i] = v[i]   
                 
        def py_gun_inplace(t,x,res):
            for i in range(ndim):
                res[i] = t*x[i]   
                
        c_fun_pointer = choreo.segm.ODE.nb_jit_c_fun_pointer(py_fun_inplace)
        c_gun_pointer = choreo.segm.ODE.nb_jit_c_fun_pointer(py_gun_inplace)
        
        c_fun_memoryview = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_ty_c_fun_memoryview")
        c_gun_memoryview = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_ty_c_gun_memoryview")
        
        c_fun_memoryview_vec = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_ty_c_fun_memoryview_vec")
        c_gun_memoryview_vec = scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "ypp_eq_ty_c_gun_memoryview_vec")

    elif eq_name.startswith("choreo_"): 

        sol_name = eq_name.removeprefix("choreo_")
        
        NBS, params_buf = load_from_solution_file(sol_name)
        NBS.ForceGreaterNStore = True
        
        reg_x0 = np.ascontiguousarray(NBS.params_to_segmpos(params_buf).swapaxes(0, 1).reshape(NBS.segm_store,-1))
        reg_v0 = np.ascontiguousarray(NBS.params_to_segmmom(params_buf).swapaxes(0, 1).reshape(NBS.segm_store,-1))
        
        t_span = (0., 1. / NBS.nint_min)
        
        x0 = reg_x0[0,:].copy()
        v0 = reg_v0[0,:].copy()
        
        ex_sol_x = reg_x0[-1,:].copy()
        ex_sol_v = reg_v0[-1,:].copy()

        NoSym = NBS.BinSpaceRotIsId.all()
        user_data = NBS.Get_ODE_params()
        
        py_fun = NBS.Compute_velocities
        py_gun = NBS.Compute_forces_nosym if NoSym else NBS.Compute_forces
        
        py_fun_vec = NBS.Compute_velocities_vectorized
        py_gun_vec = NBS.Compute_forces_vectorized_nosym if NoSym else NBS.Compute_forces_vectorized
        
        c_fun_memoryview = scipy.LowLevelCallable.from_cython(
            choreo.cython._NBodySyst                    ,
            "Compute_velocities_user_data"              ,
            user_data                                   ,
        )
        c_gun_memoryview = scipy.LowLevelCallable.from_cython(
            choreo.cython._NBodySyst                ,
            "Compute_forces_nosym_user_data"        ,
            user_data                               ,
        ) if NoSym else scipy.LowLevelCallable.from_cython(
            choreo.cython._NBodySyst                ,
            "Compute_forces_user_data"              ,
            user_data                               ,
        )
        
        c_fun_memoryview_vec = scipy.LowLevelCallable.from_cython(
            choreo.cython._NBodySyst                    ,
            "Compute_velocities_vectorized_user_data"   ,
            user_data                                   ,
        )
        c_gun_memoryview_vec = scipy.LowLevelCallable.from_cython(
            choreo.cython._NBodySyst                    ,
            "Compute_forces_vectorized_nosym_user_data" ,
            user_data                                   ,
        ) if NoSym else scipy.LowLevelCallable.from_cython(
            choreo.cython._NBodySyst                    ,
            "Compute_forces_vectorized_user_data"       ,
            user_data                                   ,
        )

    else:
        raise ValueError(f'Unknown {eq_name = }')
    
    loc = locals()
    fgun = {}

    py_fun = loc.get("py_fun")
    py_gun = loc.get("py_gun")
    if py_fun is not None and py_gun is not None:
        fgun[("py_fun", False)] = (py_fun, py_gun)    
        
    py_fun_vec = loc.get("py_fun_vec")
    py_gun_vec = loc.get("py_gun_vec")
    if py_fun_vec is not None and py_gun_vec is not None:
        fgun[("py_fun", True)] = (py_fun_vec, py_gun_vec)  
          
    c_fun_memoryview = loc.get("c_fun_memoryview")
    c_gun_memoryview = loc.get("c_gun_memoryview")
    if c_fun_memoryview is not None and c_gun_memoryview is not None:
        fgun[("c_fun_memoryview", False)] = (c_fun_memoryview, c_gun_memoryview)    
        
    c_fun_memoryview_vec = loc.get("c_fun_memoryview_vec")
    c_gun_memoryview_vec = loc.get("c_gun_memoryview_vec")
    if c_fun_memoryview_vec is not None and c_gun_memoryview_vec is not None:
        fgun[("c_fun_memoryview", True)] = (c_fun_memoryview_vec, c_gun_memoryview_vec)  
          
    c_fun_pointer = loc.get("c_fun_pointer")
    c_gun_pointer = loc.get("c_gun_pointer")
    if c_fun_pointer is not None and c_gun_pointer is not None:
        fgun[("c_fun_pointer", False)] = (c_fun_pointer, c_gun_pointer)    
              
    c_fun_pointer_vec = loc.get("c_fun_pointer_vec")
    c_gun_pointer_vec = loc.get("c_gun_pointer_vec")
    if c_fun_pointer_vec is not None and c_gun_pointer_vec is not None:
        fgun[("py_fun", True)] = (c_fun_pointer_vec, c_gun_pointer_vec)

    res = {"fgun":fgun}
    
    for key in ["nint", "t_span", "ex_sol_fun_x",  "ex_sol_fun_v", "x0", "v0", "ex_sol_x", "ex_sol_v", "reg_x0", "reg_v0"]:
        # Facultative keys
        res[key] = loc.get(key)
        
    if res.get("x0") is None:
        res["x0"] = ex_sol_fun_x(t_span[0])
    if res.get("v0") is None:
        res["v0"] = ex_sol_fun_v(t_span[0])
    if res.get("ex_sol_x") is None:
        res["ex_sol_x"] = ex_sol_fun_x(t_span[1])
    if res.get("ex_sol_v") is None:
        res["ex_sol_v"] = ex_sol_fun_v(t_span[1])

    return res
    
    
            