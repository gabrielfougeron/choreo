import functools
import inspect
import pytest

from . import test_config_precision
from . import test_config_quad_ODE
from . import test_config_NBodySyst

all_fixture_modules = [test_config_precision, test_config_quad_ODE, test_config_NBodySyst]

def ProbabilisticTest(RepeatOnFail = 10):

    def decorator(test):

        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            
            try:
                return test(*args, **kwargs)

            except AssertionError:
                
                out_str = f"Probabilistic test failed. Running test again {RepeatOnFail} times."
                head_foot = '='*len(out_str)

                print('')
                print(head_foot)
                print(out_str)
                print(head_foot)
                print('')

                for _ in range(RepeatOnFail):
                    res = test(*args, **kwargs)

                return res

        return wrapper
    
    return decorator

def RepeatTest(n = 10):

    def decorator(test):

        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            
            for _ in range(n):
                    res = test(*args, **kwargs)

            return res

        return wrapper
    
    return decorator

def RetryTest(n = 10):

    def decorator(test):

        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            
            n_fail = 0
            
            for _ in range(n):
                try:
                    res = test(*args, **kwargs)
                    break
                except AssertionError:
                    
                    n_fail += 1
                    out_str = f"Test failed {n_fail} / {n} time{"" if n==1 else "s"}."
                    head_foot = '='*len(out_str)

                    print('')
                    print(head_foot)
                    print(out_str)
                    print(head_foot)
                    print('')

            else:
                raise ValueError(f"Test failed {n} / {n} time{"" if n==1 else "s"}.")

            return res

        return wrapper
    
    return decorator


def add_intersphinx_type(type_str):
    
    if type_str in ["str", "int", "dict", "bool", "list", "tuple", "float"]:
        res = f'python:{type_str}'
    elif type_str in ["numpy.ndarray"]:
        res = f'numpy:{type_str}'
    else:
        res = type_str
        
    return res

def ParametrizeDocstrings(test):
    
    max_len_value = 800 # Values of parametrized arguments longer that this length will not be included in docs

    @functools.wraps(test)
    def wrapper(*args, **kwargs):

        return test(*args, **kwargs)

    ptm = getattr(wrapper, "pytestmark", [])
    init_doc = getattr(wrapper, "__doc__") # if doctest is empty, then __doc__ is None whether I want it or not
    
    if init_doc is None:
        init_doc = ""
    
    param_doc = '\n\nParameters\n----------\n'
    param_ptm_doc = ''
    param_fix_doc = ''
    
    all_ptm_args = []
    
    for mark in reversed(ptm):
        
        if len(mark.args) < 2:
            continue
        
        if isinstance(mark.args[0], str):
            arg_name = mark.args[0]
            arg_list = mark.args[1]
            
            all_ptm_args.append(arg_name)
            
            elem_0 = arg_list[0]
            
            type_str = f'{type(elem_0)}'[8:-2] # get type from remove <class 'type'>
            
            IsParam = ('ParameterSet' in type_str)
            
            if IsParam:
                type_str = f'{type(elem_0.values[0])}'[8:-2]
            
            param_ptm_doc += f'{arg_name} : :class:`{add_intersphinx_type(type_str)}`\n'
            
            if IsParam:
                try:
                    arg_list_str = f'{[elem.id for elem in arg_list]}'
                except:
                    arg_list_str = f'{arg_list}'
            else:
                arg_list_str = f'{arg_list}'
                
            if len(arg_list_str) < max_len_value:
                param_ptm_doc += f'    {arg_list_str}\n'
            
        else:
        
            arg_list = mark.args[1]
            elem_0 = arg_list[0]
            elem_0_name = mark.args[0][0]
            
            for i_arg, arg_name in enumerate(mark.args[0]):
               
                all_ptm_args.append(arg_name)

                type_str = f'{type(elem_0)}'[8:-2] # get type from remove <class 'type'>
                
                IsParam = ('ParameterSet' in type_str)
            
                if IsParam:
                    type_str = f'{type(elem_0.values[i_arg])}'[8:-2]
                else:
                    type_str = f'{type(elem_0[i_arg])}'[8:-2]
                   
                param_ptm_doc += f'{arg_name} : :class:`{add_intersphinx_type(type_str)}`\n' 
                
                if IsParam:
                    
                    if i_arg == 0:
                        try:
                            Id_succeed = True
                            arg_list_str = f'{[arg.id for arg in arg_list]}'
                        except:
                            Id_succeed = False
                            arg_list_str = f'{[arg[i_arg] for arg in arg_list]}'
                    else:
                        if Id_succeed:
                            arg_list_str = f'See **{elem_0_name}** above.'
                        else:
                            arg_list_str = f'{[arg[i_arg] for arg in arg_list]}'
                            
                else:
                    arg_list_str = f'{[arg[i_arg] for arg in arg_list]}'

                if len(arg_list_str) < max_len_value:
                    param_ptm_doc += f'    {arg_list_str}\n'
                
    sig = inspect.signature(wrapper)        
    for arg_name in sig.parameters:
        
        if arg_name in all_ptm_args:
            continue
        
        for mod in all_fixture_modules:
            if arg_name in dir(mod):
                
                potential_fix = getattr(mod, arg_name)
                
                if hasattr(potential_fix, "_pytestfixturefunction"):
                    
                    param_fix_doc += f'{arg_name} : :py:func:`pytest:pytest.fixture`\n See :func:`tests.test_config.{arg_name}`.\n'
   
    wrapper.__doc__ = init_doc + param_doc + param_fix_doc + param_ptm_doc

    return wrapper

