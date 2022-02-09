
import shutil
    
from  Choreo_find import *
# import Choreo_find


# def print(*args, **kwargs):
#     """My custom print() function."""
#     # Adding new arguments to the print function signature 
#     # is probably a bad idea.
#     # Instead consider testing if custom argument keywords
#     # are present in kwargs
#     builtins.print(preprint_msg,end='')
#     return builtins.print(*args, **kwargs)


nbody = 4

store_folder = './Target_res/'
store_folder = store_folder+str(nbody)


if not(os.path.isdir(store_folder)):
    os.makedirs(store_folder)
else:
    shutil.rmtree(store_folder)
    os.makedirs(store_folder)

n_eps_min = 0
n_eps_max = 251
# n_eps_max = 501
# n_eps_max = 101

freq_vid = 25

# ~ ReverseEnd = False
ReverseEnd = True

for i_eps in range(n_eps_min,n_eps_max):

    if (i_eps == 0):
        # slow_base_filename = './data/1x4_trefoil.npy'
        slow_base_filename = './data/1x4_knot.npy'
        # slow_base_filename = './data/1x4_bite.npy'
    else:
        slow_base_filename = store_folder+'/'+str(i_eps)+'.npy'


    fast_base_filename_list = ['./data/1_lone_wolf.npy','./data/1_lone_wolf.npy'    ,'./data/1_lone_wolf.npy','./data/1_lone_wolf.npy'    ] 


    nfl = len(fast_base_filename_list)

    mass_mul = [1.,1.,1.,1.]
    nTf = [1,1,1,1]
    nbs = [1,1,1,1]
    nbf = [1,1,1,1]

    # mul_loops_ini = True
    mul_loops_ini = False
    # mul_loops_ini = np.random.random() > 1./2.

    mul_loops = [mul_loops_ini for _ in range(nfl)]


    Remove_Choreo_Sym = mul_loops
    # Remove_Choreo_Sym = [False,False]
    # Remove_Choreo_Sym = [False,False]

    # Rotate_fast_with_slow = True
    Rotate_fast_with_slow = False
    # Rotate_fast_with_slow = (np.random.random() > 1./2.)

    # Optimize_Init = True
    Optimize_Init = False

    # Randomize_Fast_Init = True
    Randomize_Fast_Init = False

    all_coeffs_slow_load = np.load(slow_base_filename)
    all_coeffs_fast_load_list = []

    nbpl=[]
    mass = []
    for i in range(nfl):
        fast_base_filename = fast_base_filename_list[i]
        all_coeffs_fast_load = np.load(fast_base_filename)

        if Remove_Choreo_Sym[i]:
            
            all_coeffs_fast_load_list_temp = []

            for ib in range(nbf[i]):
                
                tshift = (ib*1.0)/nbf[i]

                all_coeffs_fast_load_list_temp.append(Transform_Coeffs(np.identity(ndim), 1, tshift, 1, all_coeffs_fast_load))

                nbpl.append(nbs[i])
                mass.extend([mass_mul[i] for j in range(nbs[i])])

            all_coeffs_fast_load_list.append(np.concatenate(all_coeffs_fast_load_list_temp,axis=0))

        else:
            all_coeffs_fast_load_list.append( all_coeffs_fast_load)
            nbpl.append(nbs[i]*nbf[i])
            mass.extend([mass_mul[i] for j in range(nbs[i]*nbf[i])])
            
            
    
    epsmul = (m.cos(m.pi * (i_eps*1./(2*n_eps_max))))**2
    # ~ epsmul = (m.cos(m.pi * (i_eps*1./(n_eps_max-1))))**2

    mass_a = np.ones(nbody)
    # mass_b = np.array([1.,1.,100.,1.],dtype=np.float64)
    # mass_b = np.array([1.,1.,1e3,1.],dtype=np.float64)
    mass_b = np.array([1.,1.2,1.5,1.2],dtype=np.float64)

    mass_a = mass_a * (nbody / mass_a.sum())
    mass_b = mass_b * (nbody / mass_b.sum())

    mass = (epsmul)*mass_a + (1.-epsmul)*mass_b


    # mass = mass / mass.sum()


    Sym_list = []
    the_lcm = m.lcm(*nbpl)
    SymName = None
    Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

    # mass = np.ones((nbody))*mass_mul

    ibody = 0
    rot_angle = 0.
    s = -1
    st = -1

    Sym_list.append(ChoreoSym(
        LoopTarget=ibody,
        LoopSource=ibody,
        SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
        TimeRev=st,
        TimeShift=fractions.Fraction(numerator=0,denominator=1)
        ))

    MomConsImposed = True
    #     MomConsImposed = False

    Use_exact_Jacobian = True
    # Use_exact_Jacobian = False

    # Look_for_duplicates = True
    Look_for_duplicates = False

    Check_Escape = True
    # Check_Escape = False

    save_init = False
    # save_init = True

    Save_img = True
    # Save_img = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches

    nint_plot_img = 10000

    color = "body"
    # color = "loop"
    # color = "velocity"
    # color = "all"

    # Save_anim = True
    # Save_anim = False
    Save_anim = ((i_eps % freq_vid) == 0)

    vid_size = (8,8) # Image size in inches
    nint_plot_anim = 2*2*2*3*3*5 
    # nperiod_anim = 1./nbody

    try:
        the_lcm
    except NameError:
        period_div = 1.
    else:
        period_div = the_lcm

    nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # Plot_trace_anim = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    n_reconverge_it_max = 1
    # n_reconverge_it_max = 1


    # ncoeff_init = 600
    # ncoeff_init = 900
    # ncoeff_init = 1800
    
    ncoeff_init = all_coeffs_slow_load.shape[2]

    disp_scipy_opt = False
    # disp_scipy_opt = True

    Newt_err_norm_max = 1e-10
    # Newt_err_norm_max_save = Newt_err_norm_max*1000
    Newt_err_norm_max_save = 1e-5

    duplicate_eps = 1e-8

    krylov_method = 'lgmres'
    # krylov_method = 'gmres'
    # krylov_method = 'bicgstab'
    # krylov_method = 'cgs'
    # krylov_method = 'minres'

    # line_search = 'armijo'
    line_search = 'wolfe'

    gradtol_list =          [1e-3   ,1e-5   ,1e-7   ,1e-9   ,1e-11  ,1e-13  ,1e-15  ]
    inner_maxiter_list =    [30     ,50     ,60     ,70     ,80     ,100    ,100    ]
    maxiter_list =          [1000   ,1000   ,1000   ,500   ,500     ,300    ,100    ]
    outer_k_list =          [5      ,5      ,5      ,5      ,7      ,7      ,7      ]
    store_outer_Av_list =   [False  ,False  ,False  ,False  ,True   ,True   ,True   ]

    n_optim_param = len(gradtol_list)

    gradtol_max = 100*gradtol_list[n_optim_param-1]
    # foundsol_tol = 1000*gradtol_list[0]
    foundsol_tol = 1e0



    n_grad_change = 1.
    # n_grad_change = 1.5


    coeff_ampl_o=1e-16
    k_infl=1
    k_max=200
    coeff_ampl_min=1e-16



    n_opt_max = 1
    # n_opt_max = 5
    # n_opt_max = 1e10

    freq_erase_dict = 1000

    all_kwargs = Pick_Named_Args_From_Dict(Find_Choreo,dict(globals(),**locals()))






    Find_Choreo(**all_kwargs)
    
video_name = 'vid.mp4'
Images_to_video(store_folder+'/',video_name,ReverseEnd)

