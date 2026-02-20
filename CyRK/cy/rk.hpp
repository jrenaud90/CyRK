#pragma once

#include <vector>
#include "c_common.hpp"
#include "cysolver.hpp"

// ####################################################################################################################
// RK Configurations
// ####################################################################################################################
struct RKConfig : public ProblemConfig {
    using ProblemConfig::ProblemConfig;

    virtual ~RKConfig() {};
    RKConfig(
        DiffeqFuncType diffeq_ptr_,
        double t_start_,
        double t_end_,
        std::vector<double>& y0_vec_,
        std::vector<char>& args_vec_,
        std::vector<double>& t_eval_vec_,
        size_t num_extra_,
        size_t expected_size_,
        size_t max_num_steps_,
        size_t max_ram_MB_,
        PreEvalFunc pre_eval_func_,
        bool capture_dense_output_,
        bool force_retain_solver_,
        std::vector<Event>& events_vec_,
        std::vector<double>& rtols_,
        std::vector<double>& atols_,
        double max_step_size_,
        double first_step_size_);

    // RK-specific configurations
    std::vector<double> rtols = std::vector<double>(1); // Relative tolerances for each dependent variable; if only 1 is provided then it will be used for every dependent variable.
    std::vector<double> atols = std::vector<double>(1); // Absolute tolerances for each dependent variable; if only 1 is provided then it will be used for every dependent variable.
    double max_step_size      = MAX_STEP;              // Default maximum step size (0 means no limit)
    double first_step_size    = 0.0;                   // Default first step size (0 means auto-calculate)

    void update_properties(
        DiffeqFuncType diffeq_ptr_,
        double t_start_,
        double t_end_,
        std::vector<double>& y0_vec_,
        std::vector<char>& args_vec_,
        std::vector<double>& t_eval_vec_,
        size_t num_extra_,
        size_t expected_size_,
        size_t max_num_steps_,
        size_t max_ram_MB_,
        PreEvalFunc pre_eval_func_,
        bool capture_dense_output_,
        bool force_retain_solver_,
        std::vector<Event>& events_vec_,
        std::vector<double>& rtols_,
        std::vector<double>& atols_,
        double max_step_size_,
        double first_step_size_);
    void initialize() override;
    virtual void update_properties_from_config(RKConfig* new_config_ptr);
};

// ####################################################################################################################
// RK Integrators
// ####################################################################################################################
class RKSolver : public CySolverBase {

// Attributes
protected:
    // Tolerances
    // For the same reason num_y is limited, the total number of tolerances are limited.
    bool use_array_rtols = false;
    bool use_array_atols = false;
    double* rtols_ptr = nullptr;
    double* atols_ptr = nullptr;

    // Step globals
    const double error_safety    = SAFETY;
    const double min_step_factor = MIN_FACTOR;
    const double max_step_factor = MAX_FACTOR;
    size_t K_stride = 0;
    size_t K_size = 0;

    // RK constants
    size_t order = 0;
    size_t error_estimator_order = 0;
    size_t n_stages       = 0;
    size_t n_stages_p1    = 0;
    size_t len_Acols      = 0;
    size_t len_Arows      = 0;
    size_t len_C          = 0;
    size_t len_Pcols      = 0;
    size_t nstages_numy   = 0;
    double error_exponent = 0.0;
    double A_at_10        = 0.0;

    // Pointers to RK constant arrays
    const double* C_ptr      = nullptr;
    const double* A_ptr      = nullptr;
    const double* B_ptr      = nullptr;
    const double* E_ptr      = nullptr;
    const double* E3_ptr     = nullptr;
    const double* E5_ptr     = nullptr;
    const double* P_ptr      = nullptr;
    const double* D_ptr      = nullptr;
    const double* AEXTRA_ptr = nullptr;
    const double* CEXTRA_ptr = nullptr;
    double* K_ptr            = nullptr;
    double** K_ptr_index_ptr = nullptr;

    // K is not const. Its values are stored in an array that is held by this class.
    std::vector<double> K            = std::vector<double>(PRE_ALLOC_NUMY * 7);
    std::vector<double*> K_ptr_index = std::vector<double*>(PRE_ALLOC_NUMY);

    // Step size parameters
    double user_provided_first_step_size = 0.0;
    double step          = 0.0;
    double step_size     = 0.0;
    double step_size_old = 0.0;
    double max_step_size = 0.0;


// Methods
protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual double p_estimate_error() noexcept override;
    virtual void p_step_implementation() noexcept override;
    virtual void p_calc_first_step_size() noexcept override;
    virtual void p_compute_stages() noexcept;

public:
    using CySolverBase::CySolverBase;

    virtual void set_Q_order(size_t* Q_order_ptr) override;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
    virtual CyrkErrorCodes setup() override;
};


// ####################################################################################################################
// Runge - Kutta 2(3)
// ####################################################################################################################
const size_t RK23_order     = 3;
const size_t RK23_n_stages  = 3;
const size_t RK23_len_Arows = 3;
const size_t RK23_len_Acols = 3;
const size_t RK23_len_C     = 3;
const size_t RK23_len_Pcols = 3;
const size_t RK23_error_estimator_order = 2;
const double RK23_error_exponent = 1.0 / (2.0 + 1.0);  // Defined as 1 / (error_order + 1)

class RK23 : public RKSolver {

protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual void p_compute_stages() noexcept override;
    virtual double p_estimate_error() noexcept override;
public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
};


// #####################################################################################################################
// Runge - Kutta 4(5)
// #####################################################################################################################
const size_t RK45_order     = 5;
const size_t RK45_n_stages  = 6;
const size_t RK45_len_Arows = 6;
const size_t RK45_len_Acols = 5;
const size_t RK45_len_C     = 6;
const size_t RK45_len_Pcols = 4;
const size_t RK45_error_estimator_order = 4;
const double RK45_error_exponent = 1.0 / (4.0 + 1.0);  // Defined as 1 / (error_order + 1)

class RK45 : public RKSolver {

protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual void p_compute_stages() noexcept override;
    virtual double p_estimate_error() noexcept override;
public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void set_Q_array(double* Q_ptr) noexcept override;

};


// #####################################################################################################################
// Runge - Kutta DOP 8(5; 3)
// #####################################################################################################################
const size_t DOP853_order         = 8;
const size_t DOP853_n_stages      = 12;
const size_t DOP853_nEXTRA_stages = 16;
const size_t DOP853_len_Arows     = 12;
const size_t DOP853_len_Acols     = 12;
const size_t DOP853_AEXTRA_rows   = 3;
const size_t DOP853_AEXTRA_cols   = 16;
const size_t DOP853_len_C         = 12;
const size_t DOP853_len_CEXTRA    = 3;
const size_t DOP853_INTERPOLATOR_POWER    = 7;
const size_t DOP853_error_estimator_order = 7;
const double DOP853_error_exponent        = 1.0 / (7.0 + 1.0);  // Defined as 1 / (error_order + 1)

class DOP853 : public RKSolver {

protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual double p_estimate_error() noexcept override;
    virtual void p_compute_stages() noexcept override;
public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
};

