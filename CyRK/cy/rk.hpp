#pragma once

#include <vector>
#include "common.hpp"
#include "cysolver.hpp"

// ########################################################################################################################
// RK Integrators
// ########################################################################################################################

// #####################################################################################################################
// Runge - Kutta 2(3)
// #####################################################################################################################
const int RK23_METHOD_INT               = 0;
const size_t RK23_order                 = 3;
const size_t RK23_n_stages              = 3;
const size_t RK23_len_Arows             = 3;
const size_t RK23_len_Acols             = 3;
const size_t RK23_len_C                 = 3;
const size_t RK23_len_Pcols             = 3;
const size_t RK23_error_estimator_order = 2;
const double RK23_error_exponent = 1.0 / (2.0 + 1.0);  // Defined as 1 / (error_order + 1)

const double RK23_A[9] = {
    // A - Row 0
    0.0,
    0.0,
    0.0,
    // A - Row 1
    1.0 / 2.0,
    0.0,
    0.0,
    // A - Row 2
    0.0,
    3.0 / 4.0,
    0.0
};
const double* const RK23_A_ptr = &RK23_A[0];

const double RK23_B[3] = {
    2.0 / 9.0,
    1.0 / 3.0,
    4.0 / 9.0
};
const double* const RK23_B_ptr = &RK23_B[0];

const double RK23_C[3] = {
    0.0,
    1.0 / 2.0,
    3.0 / 4.0
};
const double* const RK23_C_ptr = &RK23_C[0];

const double RK23_E[4] = {
    5.0 / 72.0,
    -1.0 / 12.0,
    -1.0 / 9.0,
    1.0 / 8.0
};
const double* const RK23_E_ptr = &RK23_E[0];

// P is the transpose of the one scipy uses.
const double RK23_P[12] = {
    // Column 1
    1.0,
    0.0,
    0.0,
    0.0,

    // Column 2
    -4.0 / 3.0,
    1.0,
    4.0 / 3.0,
    -1.0,

    // Column 3
    5.0 / 9.0,
    -2.0 / 3.0,
    -8.0 / 9.0,
    1.0
};
const double* const RK23_P_ptr = &RK23_P[0];

// #####################################################################################################################
// Runge - Kutta 4(5)
// #####################################################################################################################
const int RK45_METHOD_INT               = 1;
const size_t RK45_order                 = 5;
const size_t RK45_n_stages              = 6;
const size_t RK45_len_Arows             = 6;
const size_t RK45_len_Acols             = 5;
const size_t RK45_len_C                 = 6;
const size_t RK45_len_Pcols             = 4;
const size_t RK45_error_estimator_order = 4;
const double RK45_error_exponent = 1.0 / (4.0 + 1.0);  // Defined as 1 / (error_order + 1)

const double RK45_A[30] = {
    // A - Row 0
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 1
    1.0 / 5.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 2
    3.0 / 40.0,
    9.0 / 40.0,
    0.0,
    0.0,
    0.0,
    // A - Row 3
    44.0 / 45.0,
    -56.0 / 15.0,
    32.0 / 9.0,
    0.0,
    0.0,
    // A - Row 4
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
    0.0,
    // A - Row 5
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0
};
const double* const RK45_A_ptr = &RK45_A[0];

const double RK45_B[6] = {
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0
};
const double* const RK45_B_ptr = &RK45_B[0];

const double RK45_C[6] = {
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    4.0 / 5.0,
    8.0 / 9.0,
    1.0
};
const double* const RK45_C_ptr = &RK45_C[0];

const double RK45_E[7] = {
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0
};
const double* const RK45_E_ptr = &RK45_E[0];


// P is the transpose of the one scipy uses.
const double RK45_P[28] = {
    // Column 1
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,

    // Column 2
    -8048581381.0 / 2820520608.0,
    0.0,
    131558114200.0 / 32700410799.0,
    -1754552775.0 / 470086768.0,
    127303824393.0 / 49829197408.0,
    -282668133.0 / 205662961.0,
    40617522.0 / 29380423.0,

    // Column 3
    8663915743.0 / 2820520608.0,
    0.0,
    -68118460800.0 / 10900136933.0,
    14199869525.0 / 1410260304.0,
    -318862633887.0 / 49829197408.0,
    2019193451.0 / 616988883.0,
    -110615467.0 / 29380423.0,

    // Column 4
    -12715105075.0 / 11282082432.0,
    0.0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0
};
const double* const RK45_P_ptr = &RK45_P[0];


// #####################################################################################################################
// Runge - Kutta DOP 8(5; 3)
// #####################################################################################################################
const int DOP853_METHOD_INT               = 2;
const size_t DOP853_order                 = 8;
const size_t DOP853_n_stages              = 12;
const size_t DOP853_nEXTRA_stages         = 16;
const size_t DOP853_A_rows                = 12;
const size_t DOP853_A_cols                = 12;
const size_t DOP853_AEXTRA_rows           = 3;
const size_t DOP853_AEXTRA_cols           = 16;
const size_t DOP853_len_C                 = 12;
const size_t DOP853_len_CEXTRA            = 3;
const size_t DOP853_INTERPOLATOR_POWER    = 7;
const size_t DOP853_error_estimator_order = 7;
const double DOP853_error_exponent              = 1.0 / (7.0 + 1.0);  // Defined as 1 / (error_order + 1)

// Note both A and C are the _reduced_ versions.The full A and C are not shown.
const double DOP853_A[144] = {
    // A - Row 0
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 1
    5.26001519587677318785587544488e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 2
    1.97250569845378994544595329183e-2,
    5.91751709536136983633785987549e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 3
    2.95875854768068491816892993775e-2,
    0.0,
    8.87627564304205475450678981324e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 4
    2.41365134159266685502369798665e-1,
    0.0,
    -8.84549479328286085344864962717e-1,
    9.24834003261792003115737966543e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 5
    3.7037037037037037037037037037e-2,
    0.0,
    0.0,
    1.70828608729473871279604482173e-1,
    1.25467687566822425016691814123e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 6
    3.7109375e-2,
    0.0,
    0.0,
    1.70252211019544039314978060272e-1,
    6.02165389804559606850219397283e-2,
    -1.7578125e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 7
    3.70920001185047927108779319836e-2,
    0.0,
    0.0,
    1.70383925712239993810214054705e-1,
    1.07262030446373284651809199168e-1,
    -1.53194377486244017527936158236e-2,
    8.27378916381402288758473766002e-3,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 8
    6.24110958716075717114429577812e-1,
    0.0,
    0.0,
    -3.36089262944694129406857109825,
    -8.68219346841726006818189891453e-1,
    2.75920996994467083049415600797e1,
    2.01540675504778934086186788979e1,
    -4.34898841810699588477366255144e1,
    0.0,
    0.0,
    0.0,
    0.0,
    // A - Row 9
    4.77662536438264365890433908527e-1,
    0.0,
    0.0,
    -2.48811461997166764192642586468,
    -5.90290826836842996371446475743e-1,
    2.12300514481811942347288949897e1,
    1.52792336328824235832596922938e1,
    -3.32882109689848629194453265587e1,
    -2.03312017085086261358222928593e-2,
    0.0,
    0.0,
    0.0,
    // A - Row 10
    -9.3714243008598732571704021658e-1,
    0.0,
    0.0,
    5.18637242884406370830023853209,
    1.09143734899672957818500254654,
    -8.14978701074692612513997267357,
    -1.85200656599969598641566180701e1,
    2.27394870993505042818970056734e1,
    2.49360555267965238987089396762,
    -3.0467644718982195003823669022,
    0.0,
    0.0,
    // A - Row 11
    2.27331014751653820792359768449,
    0.0,
    0.0,
    -1.05344954667372501984066689879e1,
    -2.00087205822486249909675718444,
    -1.79589318631187989172765950534e1,
    2.79488845294199600508499808837e1,
    -2.85899827713502369474065508674,
    -8.87285693353062954433549289258,
    1.23605671757943030647266201528e1,
    6.43392746015763530355970484046e-1,
    0.0
};
const double* const DOP853_A_ptr = &DOP853_A[0];

// A Extra is the tranpose of SciPy's AExtra
const double DOP853_AEXTRA[48] = {  // Shape = (3, 16).T
    // Column 1
    5.61675022830479523392909219681e-2,
    3.18346481635021405060768473261e-2,
    -4.28896301583791923408573538692e-1,

    // Column 2
    0.0,
    0.0,
    0.0,

    // Column 3
    0.0,
    0.0,
    0.0,

    // Column 4
    0.0,
    0.0,
    0.0,

    // Column 5
    0.0,
    0.0,
    0.0,

    // Column 6
    0.0,
    2.83009096723667755288322961402e-2,
    -4.69762141536116384314449447206,

    // Column 7
    2.53500210216624811088794765333e-1,
    5.35419883074385676223797384372e-2,
    7.68342119606259904184240953878,

    // Column 8
    -2.46239037470802489917441475441e-1,
    -5.49237485713909884646569340306e-2,
    4.06898981839711007970213554331,

    // Column 9
    -1.24191423263816360469010140626e-1,
    0.0,
    3.56727187455281109270669543021e-1,

    // Column 10
    1.5329179827876569731206322685e-1,
    0.0,
    0.0,

    // Column 11
    8.20105229563468988491666602057e-3,
    -1.08347328697249322858509316994e-4,
    0.0,

    // Column 12
    7.56789766054569976138603589584e-3,
    3.82571090835658412954920192323e-4,
    0.0,

    // Column 13
    -8.298e-3,
    -3.40465008687404560802977114492e-4,
    -1.39902416515901462129418009734e-3,

    // Column 14
    0.0,
    1.41312443674632500278074618366e-1,
    2.9475147891527723389556272149,

    // Column 15
    0.0,
    0.0,
    -9.15095847217987001081870187138,

    // Column 16
    0.0,
    0.0,
    0.0
};
const double* const DOP853_AEXTRA_ptr = &DOP853_AEXTRA[0];

const double DOP853_CEXTRA[3] = { 0.1, 0.2, 0.777777777777777777777777777778 };
const double* const DOP853_CEXTRA_ptr = &DOP853_CEXTRA[0];

// this is stored as the tranpose of the one scipy provides
const double DOP853_D[64] = {  // shape = (interpolator power - 3 = 4, n_stages_extended = 16).T
    // Column 1
    -0.84289382761090128651353491142e+1,
    0.10427508642579134603413151009e+2,
    0.19985053242002433820987653617e+2,
    -0.25693933462703749003312586129e+2,

    // Column 2
    0.0,
    0.0,
    0.0,
    0.0,

    // Column 3
    0.0,
    0.0,
    0.0,
    0.0,

    // Column 4
    0.0,
    0.0,
    0.0,
    0.0,

    // Column 5
    0.0,
    0.0,
    0.0,
    0.0,

    // Column 6
    0.56671495351937776962531783590,
    0.24228349177525818288430175319e+3,
    -0.38703730874935176555105901742e+3,
    -0.15418974869023643374053993627e+3,

    // Column 7
    -0.30689499459498916912797304727e+1,
    0.16520045171727028198505394887e+3,
    -0.18917813819516756882830838328e+3,
    -0.23152937917604549567536039109e+3,

    // Column 8
    0.23846676565120698287728149680e+1,
    -0.37454675472269020279518312152e+3,
    0.52780815920542364900561016686e+3,
    0.35763911791061412378285349910e+3,

    // Column 9
    0.21170345824450282767155149946e+1,
    -0.22113666853125306036270938578e+2,
    -0.11573902539959630126141871134e+2,
    0.93405324183624310003907691704e+2,

    // Column 10
    -0.87139158377797299206789907490,
    0.77334326684722638389603898808e+1,
    0.68812326946963000169666922661e+1,
    -0.37458323136451633156875139351e+2,

    // Column 11
    0.22404374302607882758541771650e+1,
    -0.30674084731089398182061213626e+2,
    -0.10006050966910838403183860980e+1,
    0.10409964950896230045147246184e+3,

    // Column 12
    0.63157877876946881815570249290,
    -0.93321305264302278729567221706e+1,
    0.77771377980534432092869265740,
    0.29840293426660503123344363579e+2,

    // Column 13
    -0.88990336451333310820698117400e-1,
    0.15697238121770843886131091075e+2,
    -0.27782057523535084065932004339e+1,
    -0.43533456590011143754432175058e+2,

    // Column 14
    0.18148505520854727256656404962e+2,
    -0.31139403219565177677282850411e+2,
    -0.60196695231264120758267380846e+2,
    0.96324553959188282948394950600e+2,

    // Column 15
    -0.91946323924783554000451984436e+1,
    -0.93529243588444783865713862664e+1,
    0.84320405506677161018159903784e+2,
    -0.39177261675615439165231486172e+2,

    // Column 16
    -0.44360363875948939664310572000e+1,
    0.35816841486394083752465898540e+2,
    0.11992291136182789328035130030e+2,
    -0.14972683625798562581422125276e+3
};

const double* const DOP853_D_ptr = &DOP853_D[0];

// Note: B is equal to the 13th row of the expanded version of A(which we do not define above)
const double DOP853_B[12] = {
    5.42937341165687622380535766363e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    3.1116436695781989440891606237e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2
};
const double* const DOP853_B_ptr = &DOP853_B[0];


// Note this is the reduced C array.The expanded version is not shown.
const double DOP853_C[12] = {
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
    1.0
};
const double* const DOP853_C_ptr = &DOP853_C[0];

// All except last value equals B(B length is one less than E3).
const double DOP853_E3[13] = {
    5.42937341165687622380535766363e-2 - 0.244094488188976377952755905512,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    3.1116436695781989440891606237e-1 - 0.733846688281611857341361741547,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2 - 0.220588235294117647058823529412e-1,
    0.0
};
const double* const DOP853_E3_ptr = &DOP853_E3[0];

const double DOP853_E5[13] = {
    0.1312004499419488073250102996e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753e+1,
    -0.4957589496572501915214079952,
    0.1664377182454986536961530415e+1,
    -0.3503288487499736816886487290,
    0.3341791187130174790297318841,
    0.8192320648511571246570742613e-1,
    -0.2235530786388629525884427845e-1,
    0.
};
const double* const DOP853_E5_ptr = &DOP853_E5[0];

// ########################################################################################################################
// Classes
// ########################################################################################################################
class RKSolver : public CySolverBase {

// Attributes
protected:
    // Step globals
    const double error_safety    = SAFETY;
    const double min_step_factor = MIN_FACTOR;
    const double max_step_factor = MAX_FACTOR;

    // RK constants
    size_t order = 0;
    size_t error_estimator_order = 0;
    size_t n_stages       = 0;
    size_t n_stages_p1    = 0;
    size_t len_Acols      = 0;
    size_t len_C          = 0;
    size_t len_Pcols      = 0;
    size_t nstages_numy   = 0;
    double error_exponent = 0.0;

    // Pointers to RK constant arrays
    const double* C_ptr  = nullptr;
    const double* A_ptr  = nullptr;
    const double* B_ptr  = nullptr;
    const double* E_ptr  = nullptr;
    const double* E3_ptr = nullptr;
    const double* E5_ptr = nullptr;
    const double* P_ptr  = nullptr;
    const double* D_ptr  = nullptr;
    double* K_ptr        = nullptr;

    // K is not const. Its values are stored in an array that is held by this class.
    std::vector<double> K = std::vector<double>();

    // Tolerances
    // For the same reason num_y is limited, the total number of tolerances are limited.
    std::vector<double> rtols = std::vector<double>();
    std::vector<double> atols = std::vector<double>();
    double* rtols_ptr = nullptr;
    double* atols_ptr = nullptr;
    bool use_array_rtols = false;
    bool use_array_atols = false;

    // Step size parameters
    double user_provided_first_step_size = 0.0;
    double step          = 0.0;
    double step_size     = 0.0;
    double step_size_old = 0.0;
    double max_step_size = 0.0;

    // Error estimate
    double error_norm = 0.0;


// Methods
protected:
    virtual void p_estimate_error() override;
    virtual void p_step_implementation() override;

public:
    RKSolver();
    RKSolver(
        // Base Class input arguments
        DiffeqFuncType diffeq_ptr,
        std::shared_ptr<CySolverResult> storage_sptr,
        const double t_start,
        const double t_end,
        const double* const y0_ptr,
        const size_t num_y,
        const size_t num_extra,
        const char* args_ptr,
        const size_t size_of_args,
        const size_t max_num_steps,
        const size_t max_ram_MB,
        const bool use_dense_output,
        const double* t_eval,
        const size_t len_t_eval,
        PreEvalFunc pre_eval_func,
        // RKSolver input arguments
        const double rtol = 1.0e-3,
        const double atol = 1.0e-6,
        const double* rtols_ptr = nullptr,
        const double* atols_ptr = nullptr,
        const double max_step_size = MAX_STEP,
        const double first_step_size = 0.0
    );
    virtual void reset() override;
    virtual void calc_first_step_size() override;
    virtual void set_Q_order(size_t* Q_order_ptr) override;
    virtual void set_Q_array(double* Q_ptr) override;
};



class RK23 : public RKSolver {

protected:

public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void reset() override;
};

class RK45 : public RKSolver {

protected:

public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void reset() override;
};

class DOP853 : public RKSolver {

protected:

    std::vector<double> K_extended      = std::vector<double>();
    std::vector<double> temp_double_arr = std::vector<double>();

public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void reset() override;
    virtual void p_estimate_error() override;
};
