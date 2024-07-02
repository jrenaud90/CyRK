#pragma once

#include "cysolver.hpp"


// ########################################################################################################################
// Runge - Kutta 2(3)
// ########################################################################################################################
const int RK23_order = 3;
const size_t RK23_n_stages = 3;
const size_t RK23_len_Arows = 3;
const size_t RK23_len_Acols = 3;
const size_t RK23_len_C = 3;
const int RK23_error_estimator_order = 2;
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

const double RK23_B[3] = {
    2.0 / 9.0,
    1.0 / 3.0,
    4.0 / 9.0
};

const double RK23_C[3] = {
    0.0,
    1.0 / 2.0,
    3.0 / 4.0
};

const double RK23_E[4] = {
    5.0 / 72.0,
    -1.0 / 12.0,
    -1.0 / 9.0,
    1.0 / 8.0
};


// ########################################################################################################################
// Runge - Kutta 4(5)
// ########################################################################################################################
const int RK45_order = 5;
const size_t RK45_n_stages = 6;
const size_t RK45_len_Arows = 6;
const size_t RK45_len_Acols = 5;
const size_t RK45_len_C = 6;
const int RK45_error_estimator_order = 4;
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

const double RK45_B[6] = {
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0
};

const double RK45_C[6] = {
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    4.0 / 5.0,
    8.0 / 9.0,
    1.0
};

const double RK45_E[7] = {
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0
};


// ########################################################################################################################
// Runge - Kutta DOP 8(5; 3)
// ########################################################################################################################
const int DOP853_order = 8;
const size_t DOP853_n_stages = 12;
const size_t DOP853_A_rows = 12;
const size_t DOP853_A_cols = 12;
const size_t DOP853_len_C = 12;
const int DOP853_error_estimator_order = 7;
const double DOP853_error_exponent = 1.0 / (7.0 + 1.0);  // Defined as 1 / (error_order + 1)

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
    0.0, // Nice
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


// ########################################################################################################################
// Classes
// ########################################################################################################################
class RKSolver : public CySolverBase {

// Attributes
protected:
    // Step globals
    const double error_safety = SAFETY;
    const double min_step_factor = MIN_FACTOR;
    const double max_step_factor = MAX_FACTOR;

    // RK constants
    int order = 0;
    int error_estimator_order = 0;
    double error_exponent = 0.0;
    size_t n_stages = 0;
    size_t n_stages_p1 = 0;
    size_t len_Acols = 0;
    size_t len_C = 0;
    size_t nstages_numy = 0;

    // Pointers to RK constant arrays
    const double* C_ptr = nullptr;
    const double* A_ptr = nullptr;
    const double* B_ptr = nullptr;
    const double* E_ptr = nullptr;
    const double* E3_ptr = nullptr;
    const double* E5_ptr = nullptr;
    const double* P_ptr = nullptr;
    const double* D_ptr = nullptr;
    double* K_ptr = &this->K[0];

    // K is not const. Its values are stored in an array that is held by this class.
    double K[1] = { std::nan("") };

    // Tolerances
    // For the same reason num_y is limited, the total number of tolerances are limited.
    double rtols[25] = { std::nan("") };
    double atols[25] = { std::nan("") };
    double* rtols_ptr = &rtols[0];
    double* atols_ptr = &atols[0];
    bool use_array_rtols = false;
    bool use_array_atols = false;

    // Step size parameters
    double step = 0.0;
    double step_size_old = 0.0;
    double step_size = 0.0;
    double max_step_size = 0.0;
    bool user_provided_first_step_size = false;

    // Error estimate
    double error_norm = 0.0;


// Methods
protected:
    virtual void p_estimate_error();
    virtual void p_step_implementation() override;

public:
    RKSolver();
    virtual ~RKSolver() override;
    RKSolver(
        // Input variables
        DiffeqFuncType diffeq_ptr,
        std::shared_ptr<CySolverResult> storage_ptr,
        const double t_start,
        const double t_end,
        double* y0_ptr,
        size_t num_y,
        bool capture_extra = false,
        size_t num_extra = 0,
        double* args_ptr = nullptr,
        size_t max_num_steps = 0,
        size_t max_ram_MB = 2000,
        double rtol = 1.0e-3,
        double atol = 1.0e-6,
        double* rtols_ptr = nullptr,
        double* atols_ptr = nullptr,
        double max_step_size = MAX_STEP,
        double first_step_size = 0.0
    );
    virtual void reset() override;
    void calc_first_step_size();
};



class RK23 : public RKSolver {

protected:
    double K[4 * 25] = { 0.0 };

public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void reset() override;
};


class RK45 : public RKSolver {

protected:
    double K[7 * 25] = { 0.0 };

public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void reset() override;
};

class DOP853 : public RKSolver {

protected:
    double K[13 * 25] = { 0.0 };

public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void reset() override;
    virtual void p_estimate_error() override;
};
