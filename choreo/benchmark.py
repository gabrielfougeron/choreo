import os
import numpy as np
import timeit
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

def run_benchmark(
    all_sizes               ,
    all_funs                ,
    setup = None            ,
    n_repeat = 1            ,
    time_per_test = 0.2     ,
    timings_filename = None ,
    show = False            ,
    **show_kwargs           ,
):
    
    if timings_filename is None:
        Load_timings_file = False
        Save_timings_file = False
    else:
        Load_timings_file =  os.path.isfile(timings_filename)
        Save_timings_file = True


    n_sizes = len(all_sizes)
    n_funs = len(all_funs)

    if Load_timings_file:

        all_times = np.load(timings_filename)

        BenchmarkUpToDate = True
        BenchmarkUpToDate = BenchmarkUpToDate and (all_times.shape[0] == n_sizes  )
        BenchmarkUpToDate = BenchmarkUpToDate and (all_times.shape[1] == n_funs   )
        BenchmarkUpToDate = BenchmarkUpToDate and (all_times.shape[2] == n_repeat )

        DoBenchmark = not(BenchmarkUpToDate)
        
    else:

        DoBenchmark = True

    if DoBenchmark:

        all_times = np.zeros((n_sizes,n_funs,n_repeat))

        for i_size, size in enumerate(all_sizes):
            for i_fun, fun in enumerate(all_funs):

                code = f'all_funs[{i_fun}](x)'

                setup_vars = setup(size)

                global_dict = {'all_funs' : all_funs}
                for var, name in setup_vars:
                    global_dict[name] = var

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

                except Exception:
                    pass

        if Save_timings_file:
            np.save(timings_filename, all_times)

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
    all_funs                ,
    all_x_scalings = None   ,
    all_y_scalings = None   ,
    n_repeat = 1            ,
    color_list = None       ,
    log_plot = True         ,
    show = False            ,
    fig = None              ,
    ax = None               ,
    title = None            ,
):
    
    n_sizes = len(all_sizes)
    n_funs = len(all_funs)

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

    if log_plot:
        ax.set_yscale('log')
        ax.set_xscale('log')

    leg_patch = []
    for i_fun in range(n_funs):

        if (np.linalg.norm(all_times[:, i_fun, :]) > 0):

            leg_patch.append(
                mpl.patches.Patch(
                    color = color_list[i_fun]           ,
                    label = all_funs[i_fun].__name__    ,
                    # linestyle = linestyle       ,
                )
            )

        for i_repeat in range(n_repeat):

            plot_y_val = all_times[:, i_fun, i_repeat] / all_y_scalings[i_fun]
            plot_x_val = all_sizes * all_x_scalings[i_fun]

            if (np.linalg.norm(plot_y_val) > 0):

                ax.plot(plot_x_val, plot_y_val, color = color_list[i_fun])

    ax.legend(
        handles=leg_patch,    
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
    )

    ax.grid(True, which="major", linestyle="-")
    ax.grid(True, which="minor", linestyle="dotted")


    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()