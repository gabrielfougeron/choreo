import numpy as np
import scipy
import choreo

Small_orders = list(range(2,11))
    
ClassicalImplicitRKMethods = [
    "Gauss"         ,
    "Radau_IA"      ,
    "Radau_IIA"     ,
    "Radau_IB"      ,
    "Radau_IIB"     ,
    "Lobatto_IIIA"  ,
    "Lobatto_IIIB"  ,
    "Lobatto_IIIC"  ,
    "Lobatto_IIIC*" ,
    "Lobatto_IIID"  ,            
    "Lobatto_IIIS"  ,     
]

SymplecticImplicitRKMethodPairs = [
    ("Gauss"        , "Gauss"           ),
    ("Radau_IB"     , "Radau_IB"        ),
    ("Radau_IIB"    , "Radau_IIB"       ),
    ("Lobatto_IIID" , "Lobatto_IIID"    ),
    ("Lobatto_IIIS" , "Lobatto_IIIS"    ),
    ("Lobatto_IIIA" , "Lobatto_IIIB"    ),
    ("Lobatto_IIIC" , "Lobatto_IIIC*"   ),
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
    "SymplecticEuler"   : choreo.scipy_plus.precomputed_tables.SymplecticEuler  ,
    "StormerVerlet"     : choreo.scipy_plus.precomputed_tables.StormerVerlet    ,
    "McAte2"            : choreo.scipy_plus.precomputed_tables.McAte2           ,
    "Ruth3"             : choreo.scipy_plus.precomputed_tables.Ruth3            ,
    "McAte3"            : choreo.scipy_plus.precomputed_tables.McAte3           ,
    "Ruth4"             : choreo.scipy_plus.precomputed_tables.Ruth4            ,
    "Ruth4Rat"          : choreo.scipy_plus.precomputed_tables.Ruth4Rat         ,
    "McAte4"            : choreo.scipy_plus.precomputed_tables.McAte4           ,
    "CalvoSanz4"        : choreo.scipy_plus.precomputed_tables.CalvoSanz4       ,
    "McAte5"            : choreo.scipy_plus.precomputed_tables.McAte5           ,
    "Yoshida6A"         : choreo.scipy_plus.precomputed_tables.Yoshida6A        ,
    "Yoshida6B"         : choreo.scipy_plus.precomputed_tables.Yoshida6B        ,
    "Yoshida6C"         : choreo.scipy_plus.precomputed_tables.Yoshida6C        ,
    "KahanLi6"          : choreo.scipy_plus.precomputed_tables.KahanLi6         ,
    "McAte8"            : choreo.scipy_plus.precomputed_tables.McAte8           ,
    "KahanLi8"          : choreo.scipy_plus.precomputed_tables.KahanLi8         ,
    "Yoshida8A"         : choreo.scipy_plus.precomputed_tables.Yoshida8A        ,
    "Yoshida8B"         : choreo.scipy_plus.precomputed_tables.Yoshida8B        ,
    "Yoshida8C"         : choreo.scipy_plus.precomputed_tables.Yoshida8C        ,
    "Yoshida8D"         : choreo.scipy_plus.precomputed_tables.Yoshida8D        ,
    "Yoshida8E"         : choreo.scipy_plus.precomputed_tables.Yoshida8E        ,
    "SofSpa10"          : choreo.scipy_plus.precomputed_tables.SofSpa10         ,
}

all_fun_types = [
    "py_fun"            ,
    "c_fun_memoryview"  ,
    "c_fun_pointer"     ,
]

all_eq_names = [
    "ypp=minus_y"       ,
]

def define_ODE_ivp(eq_name):
        
    if eq_name == "ypp=minus_y":

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

        ex_sol_x = lambda t : np.array([ np.cos(t), np.sin(t)])
        ex_sol_v = lambda t : np.array([-np.sin(t), np.cos(t)])

        def py_fun(t,v):
            return np.asarray(v)
        
        def py_gun(t,x):
            return -np.asarray(x)
        
        def py_fun_inplace(t,v,res):
            for i in range(ndim):
                res[i] = v[i]   
                 
        def py_gun_inplace(t,x,res):
            for i in range(ndim):
                res[i] = -x[i]   
        
        c_fun_pointer = choreo.scipy_plus.ODE.nb_jit_c_fun_pointer(py_fun_inplace)
        c_gun_pointer = choreo.scipy_plus.ODE.nb_jit_c_fun_pointer(py_gun_inplace)
        
        c_fun_memoryview = scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "ypp_eq_my_c_fun_memoryview")
        c_gun_memoryview = scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "ypp_eq_my_c_gun_memoryview")
        
        c_fun_memoryview_vec = scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "ypp_eq_my_c_fun_memoryview_vec")
        c_gun_memoryview_vec = scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "ypp_eq_my_c_gun_memoryview_vec")
        
        return {
            "nint" : nint                   ,
            "t_span" : t_span               ,
            "ex_sol_x" : ex_sol_x           ,
            "ex_sol_v" : ex_sol_v           ,
            "fgun" : {
                ("py_fun", False) : (py_fun, py_gun)                                        ,
                ("py_fun", True ) : (py_fun, py_gun)                                        ,
                ("c_fun_memoryview", False ) : (c_fun_memoryview, c_gun_memoryview)         ,
                ("c_fun_memoryview_vec", True  ) : (c_fun_memoryview_vec, c_gun_memoryview_vec) ,
                ("c_fun_pointer", False) : (c_fun_pointer, c_gun_pointer)                   ,
                # ("c_fun_pointer", True ) : (nb_c_fun_pointer, nb_c_gun_pointer) ,
            }                               ,
        }
            