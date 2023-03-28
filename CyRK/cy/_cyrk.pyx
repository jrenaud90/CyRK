# distutils: language = c++
import cython
import numpy as np
cimport numpy as np
np.import_array()
from libcpp cimport bool as bool_cpp_t
from libc.math cimport sqrt, fabs

from CyRK.array.interp cimport interp_array, interp_complex_array

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double cabs(double complex value) nogil:
    """ Absolute value function for complex-valued inputs.
    
    Parameters
    ----------
    value : float (double complex)
        Complex-valued number.
         
    Returns
    -------
    value_abs : float (double)
        Absolute value of `value`.
    """

    cdef double v_real
    cdef double v_imag
    v_real = value.real
    v_imag = value.imag

    return sqrt(v_real * v_real + v_imag * v_imag)

# Define fused type to handle both float and complex-valued versions of y and dydt.
ctypedef fused double_numeric:
    double
    double complex

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double dabs(double_numeric value) nogil:
    """ Absolute value function for either float or complex-valued inputs.
    
    Checks the type of value and either utilizes `cabs` (for double complex) or `fabs` (for floats).
    
    Parameters
    ----------
    value : float (double_numeric)
        Float or complex-valued number.

    Returns
    -------
    value_abs : float (double)
        Absolute value of `value`.
    """

    # Check the type of value
    if double_numeric is cython.doublecomplex:
        return cabs(value)
    else:
        return fabs(value)

# # Integration Constants
# Multiply steps computed from asymptotic behaviour of errors by this.
cdef double SAFETY = 0.9
cdef double MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
cdef double MAX_FACTOR = 10.  # Maximum allowed increase in a step size.
cdef double MAX_STEP = np.inf
cdef double INF = np.inf
cdef double EPS = np.finfo(np.float64).eps
cdef double EPS_10 = EPS * 10.
cdef double EPS_100 = EPS * 100.

# RK23 Constants
cdef double RK23_C[3]
cdef double RK23_B[3]
cdef double RK23_E[4]
cdef double RK23_A[3][3]
cdef unsigned char RK23_order = 3
cdef unsigned char RK23_error_order = 2
cdef unsigned char RK23_n_stages = 3
cdef unsigned char RK23_LEN_C = 3
cdef unsigned char RK23_LEN_B = 3
cdef unsigned char RK23_LEN_E = 4
cdef unsigned char RK23_LEN_E3 = 4
cdef unsigned char RK23_LEN_E5 = 4
cdef unsigned char RK23_LEN_A0 = 3
cdef unsigned char RK23_LEN_A1 = 3

RK23_C[:] = [0, 1 / 2, 3 / 4]
RK23_B[:] = [2 / 9, 1 / 3, 4 / 9]
RK23_E[:] = [5 / 72, -1 / 12, -1 / 9, 1 / 8]

RK23_A[0][:] = [0, 0, 0]
RK23_A[1][:] = [1 / 2, 0, 0]
RK23_A[2][:] = [0, 3 / 4, 0]

# RK45 Constants
cdef double RK45_C[6]
cdef double RK45_B[6]
cdef double RK45_E[7]
cdef double RK45_A[6][5]
cdef unsigned char RK45_order = 5
cdef unsigned char RK45_error_order = 4
cdef unsigned char RK45_n_stages = 6
cdef unsigned char RK45_LEN_C = 6
cdef unsigned char RK45_LEN_B = 6
cdef unsigned char RK45_LEN_E = 7
cdef unsigned char RK45_LEN_E3 = 7
cdef unsigned char RK45_LEN_E5 = 7
cdef unsigned char RK45_LEN_A0 = 6
cdef unsigned char RK45_LEN_A1 = 5

RK45_C[:] = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1]
RK45_B[:] = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
RK45_E[:] = [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]

RK45_A[0][:] = [0, 0, 0, 0, 0]
RK45_A[1][:] = [1 / 5, 0, 0, 0, 0]
RK45_A[2][:] = [3 / 40, 9 / 40, 0, 0, 0]
RK45_A[3][:] = [44 / 45, -56 / 15, 32 / 9, 0, 0]
RK45_A[4][:] = [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0]
RK45_A[5][:] = [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]

# DOP863 Constants
cdef Py_ssize_t j_, i_
cdef unsigned char DOP_order = 8
cdef unsigned char DOP_error_order = 7
cdef unsigned char DOP_n_stages = 12
cdef unsigned char DOP_n_stages_extended = 16
cdef unsigned char DOP_LEN_C = 12  ## Reduced Size
cdef unsigned char DOP_LEN_B = 12
cdef unsigned char DOP_LEN_E = 13
cdef unsigned char DOP_LEN_E3 = 13
cdef unsigned char DOP_LEN_E5 = 13
cdef unsigned char DOP_LEN_A0 = 12  ## Reduced Size
cdef unsigned char DOP_LEN_A1 = 12  ## Reduced Size

cdef double DOP_C[16]
DOP_C = [
    0.0,
    0.526001519587677318785587544488e-01,
    0.789002279381515978178381316732e-01,
    0.118350341907227396726757197510,
    0.281649658092772603273242802490,
    0.333333333333333333333333333333,
    0.25,
    0.307692307692307692307692307692,
    0.651282051282051282051282051282,
    0.6,
    0.857142857142857142857142857142,
    1.0,
    1.0,
    0.1,
    0.2,
    0.777777777777777777777777777778]

cdef double DOP_C_REDUCED[12]
for i_ in range(12):
    DOP_C_REDUCED[i_] = DOP_C[i_]

cdef double DOP_A[16][16]
for j_ in range(16):
    for i_ in range(16):
        DOP_A[i_][j_] = 0.

DOP_A[1][0] = 5.26001519587677318785587544488e-2

DOP_A[2][0] = 1.97250569845378994544595329183e-2
DOP_A[2][1] = 5.91751709536136983633785987549e-2

DOP_A[3][0] = 2.95875854768068491816892993775e-2
DOP_A[3][2] = 8.87627564304205475450678981324e-2

DOP_A[4][0] = 2.41365134159266685502369798665e-1
DOP_A[4][2] = -8.84549479328286085344864962717e-1
DOP_A[4][3] = 9.24834003261792003115737966543e-1

DOP_A[5][0] = 3.7037037037037037037037037037e-2
DOP_A[5][3] = 1.70828608729473871279604482173e-1
DOP_A[5][4] = 1.25467687566822425016691814123e-1

DOP_A[6][0] = 3.7109375e-2
DOP_A[6][3] = 1.70252211019544039314978060272e-1
DOP_A[6][4] = 6.02165389804559606850219397283e-2
DOP_A[6][5] = -1.7578125e-2

DOP_A[7][0] = 3.70920001185047927108779319836e-2
DOP_A[7][3] = 1.70383925712239993810214054705e-1
DOP_A[7][4] = 1.07262030446373284651809199168e-1
DOP_A[7][5] = -1.53194377486244017527936158236e-2
DOP_A[7][6] = 8.27378916381402288758473766002e-3

DOP_A[8][0] = 6.24110958716075717114429577812e-1
DOP_A[8][3] = -3.36089262944694129406857109825
DOP_A[8][4] = -8.68219346841726006818189891453e-1
DOP_A[8][5] = 2.75920996994467083049415600797e1
DOP_A[8][6] = 2.01540675504778934086186788979e1
DOP_A[8][7] = -4.34898841810699588477366255144e1

DOP_A[9][0] = 4.77662536438264365890433908527e-1
DOP_A[9][3] = -2.48811461997166764192642586468
DOP_A[9][4] = -5.90290826836842996371446475743e-1
DOP_A[9][5] = 2.12300514481811942347288949897e1
DOP_A[9][6] = 1.52792336328824235832596922938e1
DOP_A[9][7] = -3.32882109689848629194453265587e1
DOP_A[9][8] = -2.03312017085086261358222928593e-2

DOP_A[10][0] = -9.3714243008598732571704021658e-1
DOP_A[10][3] = 5.18637242884406370830023853209
DOP_A[10][4] = 1.09143734899672957818500254654
DOP_A[10][5] = -8.14978701074692612513997267357
DOP_A[10][6] = -1.85200656599969598641566180701e1
DOP_A[10][7] = 2.27394870993505042818970056734e1
DOP_A[10][8] = 2.49360555267965238987089396762
DOP_A[10][9] = -3.0467644718982195003823669022

DOP_A[11][0] = 2.27331014751653820792359768449
DOP_A[11][3] = -1.05344954667372501984066689879e1
DOP_A[11][4] = -2.00087205822486249909675718444
DOP_A[11][5] = -1.79589318631187989172765950534e1
DOP_A[11][6] = 2.79488845294199600508499808837e1
DOP_A[11][7] = -2.85899827713502369474065508674
DOP_A[11][8] = -8.87285693353062954433549289258
DOP_A[11][9] = 1.23605671757943030647266201528e1
DOP_A[11][10] = 6.43392746015763530355970484046e-1

DOP_A[12][0] = 5.42937341165687622380535766363e-2
DOP_A[12][5] = 4.45031289275240888144113950566
DOP_A[12][6] = 1.89151789931450038304281599044
DOP_A[12][7] = -5.8012039600105847814672114227
DOP_A[12][8] = 3.1116436695781989440891606237e-1
DOP_A[12][9] = -1.52160949662516078556178806805e-1
DOP_A[12][10] = 2.01365400804030348374776537501e-1
DOP_A[12][11] = 4.47106157277725905176885569043e-2

DOP_A[13][0] = 5.61675022830479523392909219681e-2
DOP_A[13][6] = 2.53500210216624811088794765333e-1
DOP_A[13][7] = -2.46239037470802489917441475441e-1
DOP_A[13][8] = -1.24191423263816360469010140626e-1
DOP_A[13][9] = 1.5329179827876569731206322685e-1
DOP_A[13][10] = 8.20105229563468988491666602057e-3
DOP_A[13][11] = 7.56789766054569976138603589584e-3
DOP_A[13][12] = -8.298e-3

DOP_A[14][0] = 3.18346481635021405060768473261e-2
DOP_A[14][5] = 2.83009096723667755288322961402e-2
DOP_A[14][6] = 5.35419883074385676223797384372e-2
DOP_A[14][7] = -5.49237485713909884646569340306e-2
DOP_A[14][10] = -1.08347328697249322858509316994e-4
DOP_A[14][11] = 3.82571090835658412954920192323e-4
DOP_A[14][12] = -3.40465008687404560802977114492e-4
DOP_A[14][13] = 1.41312443674632500278074618366e-1

DOP_A[15][0] = -4.28896301583791923408573538692e-1
DOP_A[15][5] = -4.69762141536116384314449447206
DOP_A[15][6] = 7.68342119606259904184240953878
DOP_A[15][7] = 4.06898981839711007970213554331
DOP_A[15][8] = 3.56727187455281109270669543021e-1
DOP_A[15][12] = -1.39902416515901462129418009734e-3
DOP_A[15][13] = 2.9475147891527723389556272149
DOP_A[15][14] = -9.15095847217987001081870187138

cdef double DOP_A_REDUCED[12][12]
for j_ in range(12):
    for i_ in range(12):
        DOP_A_REDUCED[i_][j_] = DOP_A[i_][j_]

cdef double DOP_B[12]
for i_ in range(12):
    DOP_B[i_] = DOP_A[12][i_]

cdef double DOP_E3[13]
for i_ in range(13):
    if i_ == 12:
        DOP_E3[i_] = 0.
    else:
        DOP_E3[i_] = DOP_B[i_]
DOP_E3[0] -= 0.244094488188976377952755905512
DOP_E3[8] -= 0.733846688281611857341361741547
DOP_E3[11] -= 0.220588235294117647058823529412e-1


cdef double DOP_E5[13]
for i_ in range(13):
    DOP_E5[i_] = 0.
DOP_E5[0] = 0.1312004499419488073250102996e-1
DOP_E5[5] = -0.1225156446376204440720569753e+1
DOP_E5[6] = -0.4957589496572501915214079952
DOP_E5[7] = 0.1664377182454986536961530415e+1
DOP_E5[8] = -0.3503288487499736816886487290
DOP_E5[9] = 0.3341791187130174790297318841
DOP_E5[10] = 0.8192320648511571246570742613e-1
DOP_E5[11] = -0.2235530786388629525884427845e-1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cyrk_ode(
    diffeq,
    (double, double) t_span,
    double_numeric[:] y0,
    tuple args = tuple(),
    double rtol = 1.e-6,
    double atol = 1.e-8,
    double max_step = MAX_STEP,
    double first_step = 0.,
    unsigned char rk_method = 1,
    double[:] t_eval = np.empty((0,), dtype=np.float64),
    bool_cpp_t capture_extra = False,
    short num_extra = 0,
    bool_cpp_t interpolate_extra = False
    ):
    """ A Numba-safe Runge-Kutta Integrator based on Scipy's solve_ivp RK integrator.

    Parameters
    ----------
    diffeq : callable
        An njit-compiled function that defines the derivatives of the problem.
    t_span : Tuple[float, float]
        A tuple of the beginning and end of the integration domain's dependent variables.
    y0 : np.ndarray
        1D array of the initial values of the problem at t_span[0]
    args : tuple = tuple()
        Any additional arguments that are passed to dffeq.
    rtol : float = 1.e-6
        Integration relative tolerance used to determine optimal step size.
    atol : float = 1.e-8
        Integration absolute tolerance used to determine optimal step size.
    max_step : float = np.inf
        Maximum allowed step size.
    first_step : float = None
        Initial step size. If `None`, then the function will attempt to determine an appropriate initial step.
    rk_method : int = 1
        The type of RK method used for integration
            0 = RK23
            1 = RK45
            2 = DOP853
    t_eval : np.ndarray = None
        If provided, then the function will interpolate the integration results to provide them at the
            requested t-steps.
    capture_extra : bool = False
        If True, then additional output from the differential equation will be collected (but not used to determine
         integration error).
         Example:
            ```
            def diffeq(t, y, dy):
                a = ... some function of y and t.
                dy[0] = a**2 * sin(t) - y[1]
                dy[1] = a**3 * cos(t) + y[0]

                # Storing extra output in dy even though it is not part of the diffeq.
                dy[2] = a
            ```
    num_extra : int = 0
        The number of extra outputs the integrator should expect. With the previous example there is 1 extra output.
    interpolate_extra : bool = False
        If True, and if `t_eval` was provided, then the integrator will interpolate the extra output values at each
         step in `t_eval`.

    Returns
    -------
    time_domain : np.ndarray
        The final time domain. This is equal to t_eval if it was provided.
    y_results : np.ndarray
        The solution of the differential equation provided for each time_result.
    success : bool
        Final integration success flag.
    message : str
        Any integration messages, useful if success=False.

    """
    # Setup loop variables
    cdef Py_ssize_t s, i, j

    # Determine information about the differential equation based on its initial conditions
    cdef unsigned short y_size
    cdef double y_size_dbl, y_size_sqrt
    cdef bool_cpp_t y_is_complex
    y_size = y0.size
    y_is_complex = False
    y_size_dbl = <double>y_size
    y_size_sqrt = sqrt(y_size_dbl)

    # Check the type of the values in y0
    if double_numeric is cython.double:
        DTYPE = np.float64
    elif double_numeric is cython.doublecomplex:
        DTYPE = np.complex128
        y_is_complex = True
    else:
        # Cyrk only supports float64 and complex128.
        raise Exception('Unexpected type found for initial conditions (y0).')

    # Build time domain
    cdef double t_start, t_end, t_delta, t_delta_abs, direction, t_old, t_new, time_
    t_start = t_span[0]
    t_end   = t_span[1]
    t_delta = t_end - t_start
    t_delta_abs = fabs(t_delta)
    if t_delta >= 0.:
        direction = 1.
    else:
        direction = -1.

    # Pull out information on t-eval
    cdef unsigned int len_teval
    len_teval = t_eval.size

    # Set integration flags
    cdef bool_cpp_t success, step_accepted, step_rejected, step_error, run_interpolation, \
        store_extras_during_integration
    success           = False
    step_accepted     = False
    step_rejected     = False
    step_error        = False
    run_interpolation = False
    store_extras_during_integration = capture_extra
    if len_teval > 0:
        run_interpolation = True
    if run_interpolation and not interpolate_extra:
        # If y is eventually interpolated but the extra outputs are not being interpolated, then there is
        #  no point in storing the values during the integration. Turn off this functionality to save
        #  on computation.
        store_extras_during_integration = False

    # Initialize arrays that are based on y's size and type.
    y_init_step    = np.empty(y_size, dtype=DTYPE, order='C')
    y_new          = np.empty(y_size, dtype=DTYPE, order='C')
    y_old          = np.empty(y_size, dtype=DTYPE, order='C')
    dydt_new       = np.empty(y_size, dtype=DTYPE, order='C')
    dydt_old       = np.empty(y_size, dtype=DTYPE, order='C')
    dydt_init_step = np.empty(y_size, dtype=DTYPE, order='C')
    y_tmp          = np.empty(y_size, dtype=DTYPE, order='C')

    # Setup memory views for these arrays
    cdef double_numeric[:] y_init_step_view, y_new_view, y_old_view, dydt_new_view, dydt_old_view, \
        dydt_init_step_view, y_tmp_view
    y_init_step_view    = y_init_step
    y_new_view          = y_new
    y_old_view          = y_old
    dydt_new_view       = dydt_new
    dydt_old_view       = dydt_old
    dydt_init_step_view = dydt_init_step
    y_tmp_view          = y_tmp

    # Store y0 into the y arrays
    cdef double_numeric y_value
    for i in range(y_size):
        y_value = y0[i]
        y_new_view[i] = y_value
        y_old_view[i] = y_value
        y_tmp_view[i] = y_value
        y_init_step_view[i] = y_value

    # If extra output is true then the output of the diffeq will be larger than the size of y0.
    # Determine that extra size by calling the diffeq and checking its size.
    cdef unsigned short extra_start, total_size, store_loop_size
    extra_start = y_size
    total_size  = y_size + num_extra
    # Create arrays based on this total size
    diffeq_out     = np.empty(total_size, dtype=DTYPE, order='C')
    y_result_store = np.empty(total_size, dtype=DTYPE, order='C')
    y0_plus_extra  = np.empty(total_size, dtype=DTYPE, order='C')
    extra_result   = np.empty(num_extra, dtype=DTYPE, order='C')

    # Setup memory views
    cdef double_numeric[:] diffeq_out_view, y_result_store_view, y0_plus_extra_view, extra_result_view
    diffeq_out_view     = diffeq_out
    y_result_store_view = y_result_store
    y0_plus_extra_view  = y0_plus_extra
    extra_result_view   = extra_result

    # Capture the extra output for the initial condition.
    if capture_extra:
        diffeq(
            t_start,
            y_new,
            diffeq_out,
            *args
        )

        # Extract the extra output from the function output.
        for i in range(total_size):
            if i < extra_start:
                # Pull from y0
                y0_plus_extra_view[i] = y0[i]
            else:
                # Pull from extra output
                y0_plus_extra_view[i] = diffeq_out_view[i]
        if store_extras_during_integration:
            store_loop_size = total_size
        else:
            store_loop_size = y_size
    else:
        # No extra output
        store_loop_size = y_size

    y0_to_store = np.empty(store_loop_size, dtype=DTYPE, order='C')
    cdef double_numeric[:] y0_to_store_view
    y0_to_store_view = y0_to_store
    for i in range(store_loop_size):
        if store_extras_during_integration:
            y0_to_store_view[i] = y0_plus_extra_view[i]
        else:
            y0_to_store_view[i] = y0[i]

    # Create lists to store final outputs
    cdef list time_domain_list, y_results_list
    # Start storing results with the initial conditions
    time_domain_list = [t_start]
    y_results_list   = [y0_to_store]

    # # Determine RK scheme
    cdef unsigned char rk_order, error_order, rk_n_stages, rk_n_stages_plus1, rk_n_stages_extended
    cdef double error_expo, error_norm5, error_norm3, error_norm, error_norm_abs, error_denom
    cdef unsigned char len_C, len_B, len_E, len_E3, len_E5, len_A0, len_A1

    if rk_method == 0:
        # RK23 Method
        rk_order    = RK23_order
        error_order = RK23_error_order
        rk_n_stages = RK23_n_stages
        len_C       = RK23_LEN_C
        len_B       = RK23_LEN_B
        len_E       = RK23_LEN_E
        len_E3      = RK23_LEN_E3
        len_E5      = RK23_LEN_E5
        len_A0      = RK23_LEN_A0
        len_A1      = RK23_LEN_A1
    elif rk_method == 1:
        # RK45 Method
        rk_order    = RK45_order
        error_order = RK45_error_order
        rk_n_stages = RK45_n_stages
        len_C       = RK45_LEN_C
        len_B       = RK45_LEN_B
        len_E       = RK45_LEN_E
        len_E3      = RK45_LEN_E3
        len_E5      = RK45_LEN_E5
        len_A0      = RK45_LEN_A0
        len_A1      = RK45_LEN_A1
    else:
        # DOP853 Method
        rk_order    = DOP_order
        error_order = DOP_error_order
        rk_n_stages = DOP_n_stages
        len_C       = DOP_LEN_C
        len_B       = DOP_LEN_B
        len_E       = DOP_LEN_E
        len_E3      = DOP_LEN_E3
        len_E5      = DOP_LEN_E5
        len_A0      = DOP_LEN_A0
        len_A1      = DOP_LEN_A1

        rk_n_stages_extended = DOP_n_stages_extended

    rk_n_stages_plus1 = rk_n_stages + 1
    error_expo = 1. / (<double>error_order + 1.)

    # Build RK Arrays. Note that all are 1D except for A and K.
    A      = np.empty((len_A0, len_A1), dtype=DTYPE, order='C')
    B      = np.empty(len_B, dtype=DTYPE, order='C')
    C      = np.empty(len_C, dtype=np.float64, order='C')  # C is always float no matter what y0 is.
    E      = np.empty(len_E, dtype=DTYPE, order='C')
    E3     = np.empty(len_E3, dtype=DTYPE, order='C')
    E5     = np.empty(len_E5, dtype=DTYPE, order='C')
    E_tmp  = np.empty(y_size, dtype=DTYPE, order='C')
    E3_tmp = np.empty(y_size, dtype=DTYPE, order='C')
    E5_tmp = np.empty(y_size, dtype=DTYPE, order='C')
    K      = np.zeros((rk_n_stages_plus1, y_size), dtype=DTYPE, order='C')  # It is important K be initialized with 0s

    # Setup memory views.
    cdef double_numeric[:] B_view, E_view, E3_view, E5_view, E_tmp_view, E3_tmp_view, E5_tmp_view
    cdef double_numeric[:, :] A_view, K_view
    cdef double[:] C_view
    A_view      = A
    B_view      = B
    C_view      = C
    E_view      = E
    E3_view     = E3
    E5_view     = E5
    E_tmp_view  = E_tmp
    E3_tmp_view = E3_tmp
    E5_tmp_view = E5_tmp
    K_view      = K

    # Populate values based on externally defined constants.
    if rk_method == 0:
        # RK23 Method
        for i in range(len_A0):
            for j in range(len_A1):
                A_view[i, j] = RK23_A[i][j]
        for i in range(len_B):
            B_view[i] = RK23_B[i]
        for i in range(len_C):
            C_view[i] = RK23_C[i]
        for i in range(len_E):
            E_view[i] = RK23_E[i]
            # Dummy Variables, set equal to E
            E3_view[i] = RK23_E[i]
            E5_view[i] = RK23_E[i]
    elif rk_method == 1:
        # RK45 Method
        for i in range(len_A0):
            for j in range(len_A1):
                A_view[i, j] = RK45_A[i][j]
        for i in range(len_B):
            B_view[i] = RK45_B[i]
        for i in range(len_C):
            C_view[i] = RK45_C[i]
        for i in range(len_E):
            E_view[i] = RK45_E[i]
            # Dummy Variables, set equal to E
            E3_view[i] = RK45_E[i]
            E5_view[i] = RK45_E[i]
    else:
        # DOP853 Method
        for i in range(len_A0):
            for j in range(len_A1):
                A_view[i, j] = DOP_A_REDUCED[i][j]
        for i in range(len_B):
            B_view[i] = DOP_B[i]
        for i in range(len_C):
            C_view[i] = DOP_C_REDUCED[i]
        for i in range(len_E):
            E3_view[i] = DOP_E3[i]
            E5_view[i] = DOP_E5[i]
            E_view[i] = DOP_E5[i]
            # Dummy Variables, set equal to E3
            E_view[i] = DOP_E3[i]

    # # Determine integration parameters
    # Check tolerances
    if rtol < EPS_100:
        rtol = EPS_100

    #     atol_arr = np.asarray(atol, dtype=np.complex128)
    #     if atol_arr.ndim > 0 and atol_arr.shape[0] != y_size:
    #         # atol must be either the same for all y or must be provided as an array, one for each y.
    #         raise Exception

    # Initialize variables for start of integration
    diffeq(
            t_start,
            y_new,
            diffeq_out,
            *args
            )
    t_old = t_start
    t_new = t_start
    for i in range(y_size):
        dydt_new_view[i] = diffeq_out_view[i]
        dydt_old_view[i] = dydt_new_view[i]

    # # Determine size of first step.
    cdef double step_size, d0, d1, d2, d0_abs, d1_abs, d2_abs, h0, h1, scale
    if first_step == 0.:
        # Select an initial step size based on the differential equation.
        # .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
        #        Equations I: Nonstiff Problems", Sec. II.4.
        if y_size == 0:
            step_size = INF
        else:
            # Find the norm for d0 and d1
            d0 = 0.
            d1 = 0.
            for i in range(y_size):
                scale = atol + dabs(y_old_view[i]) * rtol

                d0_abs = dabs(y_old_view[i] / scale)
                d1_abs = dabs(dydt_old_view[i] / scale)
                d0 += (d0_abs * d0_abs)
                d1 += (d1_abs * d1_abs)

            d0 = sqrt(d0) / y_size_sqrt
            d1 = sqrt(d1) / y_size_sqrt

            if d0 < 1.e-5 or d1 < 1.e-5:
                h0 = 1.e-6
            else:
                h0 = 0.01 * d0 / d1

            h0_direction = h0 * direction
            t_init_step = t_old + h0_direction
            for i in range(y_size):
                y_init_step_view[i] = y_old_view[i] + h0_direction * dydt_old_view[i]

            diffeq(
                t_init_step,
                y_init_step,
                diffeq_out,
                *args
            )

            # Find the norm for d2
            d2 = 0.
            for i in range(y_size):
                dydt_init_step_view[i] = diffeq_out_view[i]

                # TODO: should/could this be `y_init_step` instead of `y_old_view`?
                scale = atol + dabs(y_old_view[i]) * rtol
                d2_abs = dabs( (dydt_init_step_view[i] - dydt_old_view[i]) / scale)
                d2 += (d2_abs * d2_abs)

            d2 = sqrt(d2) / (h0 * y_size_sqrt)

            if d1 <= 1.e-15 and d2 <= 1.e-15:
                h1 = max(1.e-6, h0 * 1.e-3)
            else:
                h1 = (0.01 / max(d1, d2))**error_expo

            step_size = min(100. * h0, h1)
    else:
        if first_step <= 0.:
            raise Exception('Error in user-provided step size: Step size must be a positive number.')
        elif first_step > t_delta_abs:
            raise Exception('Error in user-provided step size: Step size can not exceed bounds.')
        step_size = first_step

    # # Main integration loop
    cdef double min_step, step_factor, step
    cdef double c
    cdef double_numeric K_scale
    # Integrator Status Codes
    #   0  = Running
    #   -1 = Failed
    #   1  = Finished with no obvious issues
    cdef char status
    cdef unsigned int len_t
    status = 0
    len_t  = 1  # There is an initial condition provided so the time length is already 1
    while status == 0:
        if t_new == t_end or y_size == 0:
            t_old = t_end
            t_new = t_end
            status = 1
            break

        # Run RK integration step
        # Determine step size based on previous loop
        min_step = EPS_10
        # Look for over/undershoots in previous step size
        if step_size > max_step:
            step_size = max_step
        elif step_size < min_step:
            step_size = min_step

        # Determine new step size
        step_accepted = False
        step_rejected = False
        step_error    = False

        # # Step Loop
        while not step_accepted:

            if step_size < min_step:
                step_error = True
                status     = -1
                break

            # Move time forward for this particular step size
            step = step_size * direction
            t_new = t_old + step

            # Check that we are not at the end of integration with that move
            if direction * (t_new - t_end) > 0.:
                t_new = t_end

            # Correct the step if we were at the end of integration
            step = t_new - t_old
            step_size = fabs(step)

            # Calculate derivative using RK method
            for i in range(y_size):
                K_view[0, i] = dydt_old_view[i]

            for s in range(1, len_C):
                c = C_view[s]
                time_ = t_old + c * step

                # Dot Product (K, a) * step
                for j in range(s):
                    for i in range(y_size):
                        if j == 0:
                            # Initialize
                            y_tmp_view[i] = y_old_view[i]

                        y_tmp_view[i] = y_tmp_view[i] + (K_view[j, i] * A_view[s, j] * step)

                diffeq(
                    time_,
                    y_tmp,
                    diffeq_out,
                    *args
                )

                for i in range(y_size):
                    K_view[s, i] = diffeq_out_view[i]

            # Dot Product (K, B) * step
            for j in range(rk_n_stages):
                # We do not use rk_n_stages_plus1 here because we are chopping off the last row of K to match
                #  the shape of B.
                for i in range(y_size):
                    if j == 0:
                        # Initialize
                        y_new_view[i] = y_old_view[i]
                    y_new_view[i] = y_new_view[i] + (K_view[j, i] * B_view[j] * step)

            diffeq(
                t_new,
                y_new,
                diffeq_out,
                *args
            )
            for i in range(store_loop_size):
                if i < extra_start:
                    # Set diffeq results
                    dydt_new_view[i] = diffeq_out_view[i]
                else:
                    # Set extra results
                    extra_result_view[i - extra_start] = diffeq_out_view[i]

            if rk_method == 2:
                # Calculate Error for DOP853

                # Dot Product (K, E5) / scale and Dot Product (K, E3) * step / scale
                for i in range(y_size):
                    # Check how well this step performed.
                    scale = atol + max(dabs(y_old_view[i]), dabs(y_new_view[i])) * rtol

                    for j in range(rk_n_stages_plus1):
                        if j == 0:
                            # Initialize
                            E5_tmp_view[i] = 0.
                            E3_tmp_view[i] = 0.

                        elif j == rk_n_stages:
                            # Set last array of the K array.
                            K_view[j, i] = dydt_new_view[i]

                        K_scale = K_view[j, i] / scale
                        E5_tmp_view[i] = E5_tmp_view[i] + (K_scale * E5_view[j])
                        E3_tmp_view[i] = E3_tmp_view[i] + (K_scale * E3_view[j])

                # Find norms for each error
                error_norm5 = 0.
                error_norm3 = 0.

                # Perform summation
                for i in range(y_size):
                    error_norm5_abs = dabs(E5_tmp_view[i])
                    error_norm3_abs = dabs(E3_tmp_view[i])

                    error_norm5 += (error_norm5_abs * error_norm5_abs)
                    error_norm3 += (error_norm3_abs * error_norm3_abs)

                # Check if errors are zero
                if (error_norm5 == 0.) and (error_norm3 == 0.):
                    error_norm = 0.
                else:
                    error_denom = error_norm5 + 0.01 * error_norm3
                    error_norm = step_size * error_norm5 / sqrt(error_denom * y_size_dbl)

            else:
                # Calculate Error for RK23 and RK45
                error_norm = 0.
                # Dot Product (K, E) * step / scale
                for i in range(y_size):

                    # Check how well this step performed.
                    scale = atol + max(dabs(y_old_view[i]), dabs(y_new_view[i])) * rtol

                    for j in range(rk_n_stages_plus1):

                        if j == 0:
                            # Initialize
                            E_tmp_view[i] = 0.
                        elif j == rk_n_stages:
                            # Set last array of the K array.
                            K_view[j, i] = dydt_new_view[i]

                        K_scale = K_view[j, i] / scale
                        E_tmp_view[i] = E_tmp_view[i] + (K_scale * E_view[j] * step)

                    error_norm_abs = dabs(E_tmp_view[i])
                    error_norm += (error_norm_abs * error_norm_abs)
                error_norm = sqrt(error_norm) / y_size_sqrt

            if error_norm < 1.:
                # The error is low! Let's update this step for the next time loop
                if error_norm == 0.:
                    step_factor = MAX_FACTOR
                else:
                    step_factor = min(
                        MAX_FACTOR,
                        SAFETY * error_norm**-error_expo
                        )

                if step_rejected:
                    # There were problems with this step size on the previous step loop. Make sure factor does
                    #    not exasperate them.
                    step_factor = min(step_factor, 1.)

                step_size = step_size * step_factor
                step_accepted = True
            else:
                step_size = step_size * max(MIN_FACTOR, SAFETY * error_norm**-error_expo)
                step_rejected = True

        if not step_accepted:
            # Issue with step convergence
            status = -2
            break
        elif step_error:
            # Issue with step convergence
            status = -1
            break

        # End of step loop. Update the _now variables
        t_old = t_new
        for i in range(y_size):
            y_old_view[i] = y_new_view[i]
            dydt_old_view[i] = dydt_new_view[i]

        # Save data
        # If there is extra outputs then we need to store those at this timestep as well.
        for i in range(store_loop_size):
            if i < extra_start:
                # Pull from y result
                y_result_store_view[i] = y_new_view[i]
            else:
                # Pull from extra
                y_result_store_view[i] = extra_result_view[i - extra_start]

        y_results_list.append(
            y_result_store.copy()
        )
        time_domain_list.append(t_new)
        len_t += 1

    # # Clean up output.
    # Look at status of integration. Break out early if bad code.
    cdef str message
    message = 'Not Defined.'
    if status == 1:
        success = True
        message = 'Integration finished with no issue.'
    elif status == -1:
        message = 'Error in step size calculation: Required step size is less than spacing between numbers.'
    elif status < -2:
        message = 'Integration Failed.'

    # Create output arrays. To match the format that scipy follows, we will take the transpose of y.
    y_results_T = np.empty((store_loop_size, len_t), dtype=DTYPE, order='C')
    time_domain = np.empty(len_t, dtype=np.float64, order='C')

    # Create memory views.
    cdef double_numeric[:, :] y_results_T_view
    cdef double[:] time_domain_view
    y_results_T_view = y_results_T
    time_domain_view = time_domain

    # Populate values.
    if success:
        for i in range(len_t):
            time_domain_view[i] = time_domain_list[i]
            for j in range(store_loop_size):
                # To match the format that scipy follows, we will take the transpose of y.
                y_results_T_view[j, i] = y_results_list[i][j]

    # # If requested, run interpolation on output.
    cdef double_numeric[:, :] y_results_reduced_view
    cdef double_numeric[:] y_result_timeslice_view
    cdef double_numeric[:] y_interp_view
    cdef double_numeric[:] y_result_temp_view
    cdef double[:] t_eval_view
    if run_interpolation and success:
        # User only wants data at specific points.
        # The current version of this function has not implemented sicpy's dense output.
        #   Instead we use an interpolation.
        # OPT: this could be done inside the actual loop for performance gains.
        y_results_reduced       = np.empty((total_size, len_teval), dtype=DTYPE, order='C')
        y_result_timeslice      = np.empty(len_t, dtype=DTYPE, order='C')
        y_result_temp           = np.empty(len_teval, dtype=DTYPE, order='C')
        y_results_reduced_view  = y_results_reduced
        y_result_timeslice_view = y_result_timeslice
        y_result_temp_view      = y_result_temp
        t_eval_view = t_eval

        for j in range(y_size):
            # np.interp only works on 1D arrays so we must loop through each of the variables:
            # # Set timeslice equal to the time values at this y_j
            for i in range(len_t):
                y_result_timeslice_view[i] = y_results_T_view[j, i]

            # Perform numerical interpolation
            if double_numeric is cython.doublecomplex:
                y_result_temp = interp_complex_array(
                    t_eval_view,
                    time_domain,
                    y_result_timeslice_view,
                    y_result_temp_view
                    )
            else:
                interp_array(
                    t_eval_view,
                    time_domain_view,
                    y_result_timeslice_view,
                    y_result_temp_view
                    )

            # Store result.
            for i in range(len_teval):
                y_results_reduced_view[j, i] = y_result_temp_view[i]

        if capture_extra:
            # Right now if there is any extra output then it is stored at each time step used in the RK loop.
            # We have to make a choice on what to output do we, like we do with y, interpolate all of those extras?
            #  or do we use the interpolation on y to find new values.
            # The latter method is more computationally expensive (recalls the diffeq for each y) but is more accurate.
            if interpolate_extra:
                # Continue the interpolation for the extra values.
                for j in range(num_extra):
                    # np.interp only works on 1D arrays so we must loop through each of the variables:
                    # # Set timeslice equal to the time values at this y_j
                    for i in range(len_t):
                        y_result_timeslice_view[i] = y_results_T_view[extra_start + j, i]

                    # Perform numerical interpolation
                    if double_numeric is cython.doublecomplex:
                        y_result_temp = interp_complex_array(
                                t_eval_view,
                                time_domain,
                                y_result_timeslice_view,
                                y_result_temp_view
                                )
                    else:
                        interp_array(
                                t_eval_view,
                                time_domain_view,
                                y_result_timeslice_view,
                                y_result_temp_view
                                )

                    # Store result.
                    for i in range(len_teval):
                        y_results_reduced_view[extra_start + j, i] = y_result_temp_view[i]
            else:
                # Use y and t to recalculate the extra outputs
                y_interp = np.empty(y_size, dtype=DTYPE)
                y_interp_view = y_interp
                for i in range(len_teval):
                    time_ = t_eval[i]
                    for j in range(y_size):
                        y_interp_view[j] = y_results_reduced_view[j, i]

                    diffeq(
                        time_, y_interp, diffeq_out, *args
                    )

                    for j in range(num_extra):
                        y_results_reduced_view[extra_start + j, i] = diffeq_out_view[extra_start + j]

        # Replace the output y results and time domain with the new reduced one
        y_results_T = np.empty((total_size, len_teval), dtype=DTYPE, order='C')
        time_domain = np.empty(len_teval, dtype=np.float64, order='C')
        y_results_T_view = y_results_T
        time_domain_view = time_domain
        for i in range(len_teval):
            time_domain_view[i] = t_eval[i]
            for j in range(total_size):
                # To match the format that scipy follows, we will take the transpose of y.
                y_results_T_view[j, i] = y_results_reduced_view[j, i]

    return time_domain, y_results_T, success, message
