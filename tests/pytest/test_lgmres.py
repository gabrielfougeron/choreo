from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=["matplotlib","sparseqr","networkx","./choreo-0.2.0-cp310-cp310-emscripten_3_1_27_wasm32.whl"])
async def test_scipy_lgmres(selenium_standalone):

    import numpy as np
    import choreo

    # details in this functions are not important.
    # Ultimately, here is a call to an scipy.optimize.newton_krylov ==> https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton_krylov.html
    def fun(method):

        file_basename = ''

        SymName = 'C'

        LookForTarget = False

        nbpl=[5]

        SymType = []

        Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

        mass = np.ones((nbody),dtype=np.float64)

        # MomConsImposed = True
        MomConsImposed = False

        store_folder = './'

        Use_exact_Jacobian = True
        # Use_exact_Jacobian = False

        Look_for_duplicates = True
        # Look_for_duplicates = False

        Check_Escape = True
        # Check_Escape = False

        # Penalize_Escape = True
        Penalize_Escape = False

        save_first_init = False
        # save_first_init = True

        save_all_inits = False
        # save_all_inits = True

        # Save_img = True
        Save_img = False

        # Save_thumb = True
        Save_thumb = False

        img_size = (8,8) # Image size in inches
        thumb_size = (1,1) # Image size in inches
        
        color = "body"
        # color = "loop"
        # color = "velocity"
        # color = "all"

        # Save_anim = True
        Save_anim = False

        vid_size = (8,8) # Image size in inches
        nint_plot_anim = 2*2*2*3*3*5*2
        # nperiod_anim = 1./nbody
        dnint = 30

        nint_plot_img = nint_plot_anim * dnint

        try:
            the_lcm
        except NameError:
            period_div = 1.
        else:
            period_div = the_lcm

        nperiod_anim = 1.
        # nperiod_anim = 1./period_div

        Plot_trace_anim = True
        # Plot_trace_anim = False

        # Save_Newton_Error = True
        Save_Newton_Error = False

        n_reconverge_it_max = 2

        ncoeff_init = 200

        disp_scipy_opt = False
        # disp_scipy_opt = True
        
        max_norm_on_entry = 1e20

        Newt_err_norm_max = 1e-10
        Newt_err_norm_max_save = 1e-9

        duplicate_eps = 1e-8

        krylov_method = method

        line_search = 'wolfe'
        linesearch_smin = 0.01
        
        gradtol_list =          [1e-1   ,1e-3   ,1e-5   ,1e-7   ,1e-9   ,1e-11  ,1e-13  ]
        inner_maxiter_list =    [30     ,30     ,50     ,60     ,70     ,80     ,100    ]
        maxiter_list =          [100    ,1000   ,1000   ,1000   ,500    ,500    ,300    ]
        outer_k_list =          [5      ,5      ,5      ,5      ,5      ,7      ,7      ]
        store_outer_Av_list =   [False  ,False  ,False  ,False  ,False  ,True   ,True   ]
        
        n_optim_param = len(gradtol_list)
        
        gradtol_max = 100*gradtol_list[n_optim_param-1]
        # foundsol_tol = 1000*gradtol_list[0]
        foundsol_tol = 1e10

        escape_fac = 1e0
        escape_min_dist = 1
        escape_pow = 2.0

        n_grad_change = 1.
        # n_grad_change = 1.5

        coeff_ampl_o=3e-2
        # coeff_ampl_o=1e0
        k_infl=0
        k_max=10
        # k_max=200
        # k_max=200
        coeff_ampl_min=1e-16

        freq_erase_dict = 100
        hash_dict = {}

        n_opt = 0
        n_opt_max = 1
        
        n_find_max = 1

        mul_coarse_to_fine = 3

        Save_All_Coeffs = True
        # Save_All_Coeffs = False

        # Save_Init_Pos_Vel_Sol = True
        Save_Init_Pos_Vel_Sol = False


        n_save_pos = 'auto'
        Save_All_Pos = True
        # Save_All_Pos = False

        plot_extend = 0.

        all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(globals(),**locals()))
        choreo.Find_Choreo(**all_kwargs)


    # We repeat the test a few times (more likelihood of repeatability)
    n_repeat = 5

    for i in range(n_repeat):
        fun('bicgstab')  

    print('')
    print('===========================================================')
    print('No errors yet')
    print('===========================================================')
    print('')


    for i in range(n_repeat):
        fun('lgmres')  

    print('')
    print('===========================================================')
    print('An error will occur before this gets printed')
    print('===========================================================')
    print('')



