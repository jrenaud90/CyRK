#pragma once

#include <vector>
#include "c_common.hpp"
#include "cysolver.hpp"

// ####################################################################################################################
// RK Configurations
// ####################################################################################################################
/* The Runge-Kutta methods do not need any configuration beyond what `ProblemConfig` already
   carries. This subclass is retained so that existing code that builds an `RKConfig` keeps
   working and so that RK-only options have a home if they are ever needed. */
struct RKConfig : public ProblemConfig {
    using ProblemConfig::ProblemConfig;

    virtual ~RKConfig() {};
};

// ####################################################################################################################
// RK Integrators
// ####################################################################################################################
class RKSolver : public CySolverBase {

// Attributes
protected:
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
    double step_size_old = 0.0;


// Methods
protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual double p_estimate_error() noexcept override;
    virtual void p_step_implementation() noexcept override;
    virtual void p_compute_stages() noexcept;

public:
    using CySolverBase::CySolverBase;

    virtual void set_Q_order(size_t* Q_order_ptr) override;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
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

