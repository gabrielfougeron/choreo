import os
import numpy as np
import timeit
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

def isnotfinite(arr):
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

def run_benchmark(
    all_sizes               ,
    all_funs                ,
    mode = "timings"        ,
    setup = None            ,
    n_repeat = 1            ,
    time_per_test = 0.2     ,
    filename = None         ,
    ForceBenchmark = False  ,
    show = False            ,
    StopOnExcept=False      ,
    **show_kwargs           ,
):
    
    if filename is None:
        Load_timings_file = False
        Save_timings_file = False
    else:
        Load_timings_file =  os.path.isfile(filename) and not(ForceBenchmark)
        Save_timings_file = True

    n_sizes = len(all_sizes)
    
    if isinstance(all_funs,dict):
        all_funs_list = [fun for fun in all_funs.values()]
        
    else:    
        all_funs_list = [fun for fun in all_funs]
    
    n_funs = len(all_funs_list)

    if Load_timings_file:

        all_times = np.load(filename)

        BenchmarkUpToDate = True
        BenchmarkUpToDate = BenchmarkUpToDate and (all_times.shape[0] == n_sizes  )
        BenchmarkUpToDate = BenchmarkUpToDate and (all_times.shape[1] == n_funs   )
        BenchmarkUpToDate = BenchmarkUpToDate and (all_times.shape[2] == n_repeat )

        DoBenchmark = not(BenchmarkUpToDate)
        
    else:

        DoBenchmark = True

    if DoBenchmark:

        all_times = np.zeros((n_sizes,n_funs,n_repeat))

        if mode == "timings":

            for i_size, size in enumerate(all_sizes):
                for i_fun, fun in enumerate(all_funs_list):

                    setup_vars = setup(size)

                    vars_str = ''
                    global_dict = {'all_funs_list' : all_funs_list}
                    for var, name in setup_vars:
                        global_dict[name] = var
                        vars_str += name+','
                        
                    code = f'all_funs_list[{i_fun}]({vars_str})'

                    Timer = timeit.Timer(
                        code,
                        globals = global_dict,
                    )

                    try:
                        # For functions that require caching
                        Timer.timeit(number = 1)

                        # Estimate time of everything
                        n_timeit_0dot2, est_time = Timer.autorange()

                        n_timeit = math.ceil(n_timeit_0dot2 * time_per_test / est_time)

                        times = Timer.repeat(
                            repeat = n_repeat,
                            number = n_timeit,
                        )
                        
                        all_times[i_size, i_fun, :] = np.array(times) / n_timeit

                    except Exception as exc:
                        if StopOnExcept:
                            raise exc
                        
        elif mode == "scalar_output":    
            
            for i_size, size in enumerate(all_sizes):
                for i_fun, fun in enumerate(all_funs_list):
                    for i_repeat in range(n_repeat):
                        all_times[i_size, i_fun, i_repeat] = fun(setup(size))

        if Save_timings_file:
            np.save(filename, all_times)

    if show:
        plot_benchmark(
            all_times           ,
            all_sizes           ,
            all_funs            ,
            n_repeat = n_repeat ,
            show = show         ,
            **show_kwargs       ,
        )

    return all_times

def plot_benchmark(
    all_times               ,
    all_sizes               ,
    all_funs = None         ,
    all_names = None        ,
    all_xvalues = None      ,
    all_x_scalings = None   ,
    all_y_scalings = None   ,
    n_repeat = 1            ,
    color_list = None       ,
    logx_plot = None        ,
    logy_plot = None        ,
    plot_ylim = None        ,
    plot_xlim = None        ,
    clip_vals = False       ,
    stop_after_first_clip = False,
    show = False            ,
    fig = None              ,
    ax = None               ,
    title = None            ,
    transform = None        ,
):
    
    n_sizes = len(all_sizes)
    
    if isinstance(all_funs,dict):
        all_funs_list = [fun for fun in all_funs.values()]
        
    else:    
        all_funs_list = [fun for fun in all_funs]
    
    n_funs = len(all_funs_list)

    if all_names is None:
        
        all_names_list = []
        
        if isinstance(all_funs,dict):
            for name, fun in all_funs.items():
                all_names_list.append(name)
        else:
            
            for fun in all_funs_list:
                if hasattr(fun,'__name__'):
                    all_names_list.append(fun.__name__)
                else:
                    raise ValueError('Could not determine names of functions from arguments')
                
    else:
        all_names_list = [name for name in all_names]
            
    assert len(all_names_list) == n_funs
    assert all_times.shape[0] == n_sizes  
    assert all_times.shape[1] == n_funs   
    assert all_times.shape[2] == n_repeat 

    if all_x_scalings is None:
        all_x_scalings = np.ones(n_funs)
    else:
        assert all_x_scalings.shape == (n_funs,)

    if all_y_scalings is None:
        all_y_scalings = np.ones(n_funs)
    else:
        assert all_y_scalings.shape == (n_funs,)

    if (ax is None) or (fig is None):

        fig = plt.figure()  
        ax = fig.add_subplot(1,1,1)
        

    if color_list is None:
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    n_colors = len(color_list)

    if logx_plot is None:
        logx_plot = (transform is None)
        
    if logy_plot is None:
        logy_plot = (transform is None)

    if logx_plot:
        ax.set_xscale('log')
    if logy_plot:
        ax.set_yscale('log')

    if clip_vals and (plot_ylim is None):
        
        raise ValueError('Need a range to clip values')

    leg_patch = []
    for i_fun in range(n_funs):

        if (np.linalg.norm(all_times[:, i_fun, :]) > 0):
            
            color = color_list[i_fun % n_colors]
            
            leg_patch.append(
                mpl.patches.Patch(
                    color = color       ,
                    label = all_names_list[i_fun]   ,
                    # linestyle = linestyle       ,
                )
            )

            for i_repeat in range(n_repeat):

                plot_y_val = all_times[:, i_fun, i_repeat] / all_y_scalings[i_fun]
                
                if all_xvalues is None:
                    plot_x_val = all_sizes * all_x_scalings[i_fun]
                else:   
                    plot_x_val = all_xvalues[:, i_fun, i_repeat] / all_y_scalings[i_fun]
                
                if transform in ["pol_growth_order", "pol_cvgence_order"]:
                    
                    transformed_plot_y_val = np.zeros_like(plot_y_val)
                    
                    for i_size in range(1,n_sizes):
                        
                        ratio_y = plot_y_val[i_size] / plot_y_val[i_size-1]
                        ratio_x = plot_x_val[i_size] / plot_x_val[i_size-1]
                        
                        try:
                            transformed_plot_y_val[i_size] = math.log(ratio_y) / math.log(ratio_x)

                        except:
                            transformed_plot_y_val[i_size] = np.nan
                            
                    transformed_plot_y_val[0] = np.nan
                                    
                    plot_y_val = transformed_plot_y_val

                    if transform == "pol_cvgence_order":
                        plot_y_val = - transformed_plot_y_val
                    else:
                        plot_y_val = transformed_plot_y_val
                    
                if clip_vals:

                    for i_size in range(n_sizes):
                
                        if plot_y_val[i_size] < plot_ylim[0]:

                            if stop_after_first_clip:
                                for j_size in range(i_size,n_sizes):
                                    plot_y_val[j_size] = np.nan
                                break
                            else:
                                plot_y_val[i_size] = np.nan
                                
                        elif plot_y_val[i_size] > plot_ylim[1]:

                            if stop_after_first_clip:
                                for j_size in range(i_size,n_sizes):
                                    plot_y_val[j_size] = np.nan
                                break
                            else:
                                plot_y_val[i_size] = np.nan
                        
                mask = isnotfinite(plot_y_val)
                masked_plot_y_val = np.ma.array(plot_y_val, mask = mask)

                if (np.ma.max(masked_plot_y_val) > 0):
                    ax.plot(plot_x_val, plot_y_val, color = color)

    ax.legend(
        handles=leg_patch,    
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
    )

    ax.grid(True, which="major", linestyle="-")
    ax.grid(True, which="minor", linestyle="dotted")

    if plot_xlim is not None:
        ax.set_xlim(plot_xlim)
        
    if plot_ylim is not None:
        ax.set_ylim(plot_ylim)

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()