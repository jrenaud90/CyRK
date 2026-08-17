#pragma once

#include <vector>

#include "c_common.hpp"
#include "c_lsoda.hpp"
#include "cysolver.hpp"

// ####################################################################################################################
// LSODA Configurations
// ####################################################################################################################
/* Options that only apply to LSODA. Everything else (tolerances, maximum step size, first step
   size, analytic Jacobian) is inherited from `ProblemConfig`. */
struct LSODAConfig : public ProblemConfig {
    using ProblemConfig::ProblemConfig;

    virtual ~LSODAConfig() {};

    // Smallest step size that LSODA is allowed to take. Zero leaves the step size unbounded from
    // below and controlled solely by the error test.
    double min_step_size = 0.0;

    // Highest order allowed for the non-stiff (Adams) and stiff (BDF) formulas. LSODA clamps
    // these to 12 and 5 respectively.
    size_t max_order_nonstiff = 12;
    size_t max_order_stiff    = 5;

    /* Bandwidth of the Jacobian: entry (i, j) is assumed to be zero unless
       `i - num_lower <= j <= i + num_upper`. Leaving both at `MAX_SIZET_SIZE` tells LSODA to treat
       the Jacobian as dense, which is the default. Setting them is worthwhile for large banded
       systems because LSODA then needs far fewer differential equation calls to build the
       Jacobian and a much cheaper factorization. A configured analytic Jacobian must write its
       result in band storage when these are set (see `JacobianFuncType`). */
    size_t num_lower = MAX_SIZET_SIZE;
    size_t num_upper = MAX_SIZET_SIZE;

    // True if the user narrowed the Jacobian to a band.
    bool banded_jacobian() const noexcept;
    void initialize() override;
    void update_properties_from_config(ProblemConfig* new_config_ptr) override;
};

// ####################################################################################################################
// LSODA Integrator
// ####################################################################################################################
/* Adams / BDF integrator with automatic stiffness detection and switching.

   This class owns the work arrays and the persistent state of the vendored ODEPACK
   implementation in "c_lsoda.cpp" and drives it one step at a time (ODEPACK's itask = 5), which
   is the same approach that SciPy's `scipy.integrate.LSODA` takes. Doing so lets CyRK keep
   control of the solution storage, the event handling, and the `t_eval` interpolation while the
   stepping itself, the order and step size selection, and the method switching are all handled by
   the original algorithm.
*/
class LSODA : public CySolverBase {

// Attributes
protected:
    // Persistent state of the ODEPACK common blocks.
    c_lsoda_common_t common_state = c_lsoda_common_t();

    // ODEPACK work arrays.
    std::vector<double> rwork_vec = std::vector<double>(0);
    std::vector<int> iwork_vec    = std::vector<int>(0);
    double* rwork_ptr             = nullptr;
    int* iwork_ptr                = nullptr;

    // ODEPACK control flags.
    int itol          = 1;  // Which of rtol / atol are arrays.
    int istate        = 1;  // 1 on the first call, 2 afterwards, negative on failure.
    int jacobian_type = 2; // ODEPACK "jt": 1/2 for a dense Jacobian, 4/5 for a banded one.

    // Band structure of the Jacobian; only meaningful when `use_banded_jacobian` is set.
    bool use_banded_jacobian = false;
    size_t num_lower         = 0;
    size_t num_upper         = 0;

// Methods
protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual CyrkErrorCodes p_finalize_setup() noexcept override;
    // ODEPACK selects its own first step size, so the generic estimator is skipped.
    virtual void p_calc_first_step_size() noexcept override;
    virtual void p_step_implementation() noexcept override;

    // Translate an ODEPACK `istate` failure into a CyRK error code.
    static CyrkErrorCodes p_convert_istate(const int istate_) noexcept;

    // Callbacks handed to the vendored solver. `user_data` is always the owning LSODA instance.
    static void p_diffeq_hook(int* neq, double* t, double* y, double* ydot, void* user_data);
    static void p_jacobian_hook(int* neq, double* t, double* y, int* ml, int* mu, double* pd, int* nrowpd, void* user_data);

public:
    // Copy over base class constructors
    using CySolverBase::CySolverBase;

    virtual void set_Q_order(size_t* Q_order_ptr) override;
    virtual void set_Q_order_max(size_t* Q_order_max_ptr) override;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
    virtual double get_dense_step() const noexcept override;
    virtual double* get_dense_base_y_ptr() noexcept override;
};
