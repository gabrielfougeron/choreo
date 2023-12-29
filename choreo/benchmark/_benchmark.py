import os
import numpy as np
import numpy.typing
import timeit
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
import typing

def isnotfinite(arr):
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

default_color_list = list(mpl.colors.TABLEAU_COLORS)
default_color_list.append(mpl.colors.BASE_COLORS['b'])
default_color_list.append(mpl.colors.BASE_COLORS['g'])
default_color_list.append(mpl.colors.BASE_COLORS['r'])
default_color_list.append(mpl.colors.BASE_COLORS['m'])
default_color_list.append(mpl.colors.BASE_COLORS['k'])

default_linestyle_list = ['solid']

def run_benchmark(
    all_sizes       : np.typing.ArrayLike                           ,
    all_funs        : dict | typing.Iterable                        ,
    mode            : str                   = "timings"             ,
    setup           : typing.Callable[
        [int]   ,
        typing.List[
            typing.Tuple[
                typing.Any  ,
                str         ,
            ]
        ]
    ]                                       = (lambda n: [n,'n'])   ,
    n_repeat        : int                   = 1                     ,
    time_per_test   : float                 = 0.2                   ,
    filename        : str | None            = None                  ,
    ForceBenchmark  : bool                  = False                 ,
    show            : bool                  = False                 ,
    StopOnExcept    : bool                  = False                 ,
    **show_kwargs   : typing.Dict[
        str,
        typing.Any
    ]          ,
) -> np.typing.NDArray[np.float64] :
    
    """
    run_benchmark _summary_

    _extended_summary_

    Parameters
    ----------
    all_sizes : np.typing.ArrayLike
        _description_
    all_funs : dict  |  typing.Iterable
        _description_
    mode : str, optional
        _description_, by default "timings"
    setup : _type_, optional
        _description_, by default (lambda n: [n,'n'])
    n_repeat : int, optional
        _description_, by default 1
    time_per_test : float, optional
        _description_, by default 0.2
    filename : str  |  None, optional
        _description_, by default None
    ForceBenchmark : bool, optional
        _description_, by default False
    show : bool, optional
        _description_, by default False
    StopOnExcept : bool, optional
        _description_, by default False

    Returns
    -------
    np.typing.NDArray[np.float64]
        _description_

    Raises
    ------
    exc
        _description_
    ValueError
        _description_
    """    
    
    if setup is None:
        setup = lambda n: []
    
    if filename is None:
        Load_timings_file = False
        Save_timings_file = False
    else:
        Load_timings_file =  os.path.isfile(filename) and not(ForceBenchmark)
        Save_timings_file = True

    n_sizes = len(all_sizes)
    
    if isinstance(all_funs, dict):
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
                        
        else:
            
            raise ValueError(f'Unknown mode {mode}')

        if Save_timings_file:
            np.save(filename, all_times)

    if show:
        plot_benchmark(
            all_times           ,
            all_sizes           ,
            all_funs            ,
            show = show         ,
            **show_kwargs       ,
        )

    return all_times

def plot_benchmark(
    all_times               : np.typing.ArrayLike   ,
    all_sizes               : np.typing.ArrayLike   ,
    all_funs                : typing.Dict[str, callable] |
                              typing.Iterable[str] | 
                              None                              = None                  ,
    all_names               : typing.Iterable[str] | None       = None                  ,
    all_xvalues             : np.typing.ArrayLike | None        = None                  ,
    all_x_scalings          : np.typing.ArrayLike | None        = None                  ,
    all_y_scalings          : np.typing.ArrayLike | None        = None                  ,
    color_list              : list                              = default_color_list    ,
    linestyle_list          : list                              = default_linestyle_list,
    logx_plot               : bool | None                       = None                  ,
    logy_plot               : bool | None                       = None                  ,
    plot_ylim               : tuple | None                      = None                  ,
    plot_xlim               : tuple | None                      = None                  ,
    clip_vals               : bool                              = False                 ,
    stop_after_first_clip   : bool                              = False                 ,
    show                    : bool                              = False                 ,
    fig                     : matplotlib.figure.Figure | None   = None                  ,
    ax                      : plt.Axes | None                   = None                  ,
    title                   : str | None                        = None                  ,
    plot_legend             : bool                              = True                  ,
    plot_grid               : bool                              = True                  ,
    transform               : str | None                        = None                  ,
    relative_to             : np.typing.ArrayLike | None        = None                  ,
) -> None :
    """
    plot_benchmark _summary_

    _extended_summary_

    Parameters
    ----------
    all_times : np.typing.ArrayLike
        _description_
    all_sizes : np.typing.ArrayLike
        _description_
    all_funs : typing.Dict[str, callable] | typing.Iterable[str] | None, optional
        _description_, by default None
    all_names : typing.Iterable[str] | None, optional
        _description_, by default None
    all_xvalues : np.typing.ArrayLike | None, optional
        _description_, by default None
    all_x_scalings : np.typing.ArrayLike | None, optional
        _description_, by default None
    all_y_scalings : np.typing.ArrayLike | None, optional
        _description_, by default None
    color_list : list, optional
        _description_, by default default_color_list
    linestyle_list : list, optional
        _description_, by default default_linestyle_list
    logx_plot : bool | None, optional
        _description_, by default None
    logy_plot : bool | None, optional
        _description_, by default None
    plot_ylim : bool | None, optional
        _description_, by default None
    plot_xlim : bool | None, optional
        _description_, by default None
    clip_vals : bool, optional
        _description_, by default False
    stop_after_first_clip : bool, optional
        _description_, by default False
    show : bool, optional
        _description_, by default False
    fig : matplotlib.figure.Figure | None, optional
        _description_, by default None
    ax : plt.Axes | None, optional
        _description_, by default None
    title : str | None, optional
        _description_, by default None
    plot_legend : bool, optional
        _description_, by default True
    plot_grid : bool, optional
        _description_, by default True
    transform : str | None, optional
        _description_, by default None
    relative_to : np.typing.ArrayLike | None, optional
        _description_, by default None

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """    

    
    n_sizes = all_times.shape[0] 
    n_funs = all_times.shape[1]
    n_repeat = all_times.shape[2]
    
    assert n_sizes == len(all_sizes)

    if all_names is None:
        
        if all_funs is None:
            all_names_list = ['Anonymous function']*n_funs

        else:
            
            all_names_list = []
            
            if isinstance(all_funs, dict):
                for name, fun in all_funs.items():
                    all_names_list.append(name)
            
            else:    
                for fun in all_funs:
                    all_names_list.append(getattr(fun, '__name__', 'Anonymous function'))
                    
    else:
        all_names_list = [name for name in all_names]
    
    assert n_funs == len(all_names_list)

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
        
    if (relative_to is None):
        relative_to_array = np.ones((n_sizes))
        
    else:
        if isinstance(relative_to, np.ndarray):
            relative_to_array = relative_to
            assert relative_to_array.shape == (n_sizes,)
        
        else:
            
            if isinstance(relative_to, int):
                relative_to_idx = relative_to
                
            elif isinstance(relative_to, str):
                try:
                    relative_to_idx = all_names_list.index(relative_to)
                except ValueError:
                    raise ValueError(f'{relative_to} is not a known name')
            else:
                raise ValueError(f'Invalid relative_to argument {relative_to}')
            
            assert np.all(all_times[:, relative_to_idx, :] > 0.)
            
            relative_to_array = (np.sum(all_times[:, relative_to_idx, :], axis=1) / n_repeat)
        
    n_colors = len(color_list)
    n_linestyle = len(linestyle_list)

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
            linestyle = linestyle_list[i_fun % n_linestyle]
            
            leg_patch.append(
                mpl.patches.Patch(
                    color = color                   ,
                    label = all_names_list[i_fun]   ,
                    linestyle = linestyle           ,
                )
            )

            for i_repeat in range(n_repeat):

                plot_y_val = all_times[:, i_fun, i_repeat] / relative_to_array # Broadcast
                plot_y_val /= all_y_scalings[i_fun]
                
                if all_xvalues is None:
                    plot_x_val = all_sizes * all_x_scalings[i_fun]
                else:   
                    plot_x_val = all_xvalues[:, i_fun, i_repeat] / all_x_scalings[i_fun]
                
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
                    ax.plot(
                        plot_x_val              ,
                        plot_y_val              ,
                        color = color           ,
                        linestyle = linestyle   ,
                    )

    if plot_legend:
        ax.legend(
            handles=leg_patch,    
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
        )

    if plot_grid:
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