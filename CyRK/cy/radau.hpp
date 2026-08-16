#pragma once

#include <complex>
#include <vector>

#include "c_common.hpp"
#include "cysolver.hpp"

// ####################################################################################################################
// Radau Integrator
// ####################################################################################################################
/* Implicit Runge-Kutta integrator of the Radau IIA family of order 5.

   Unlike BDF and LSODA this is a single-step method, so it carries no solution history and handles
   a change of step size without rescaling anything. That makes it a good choice for a stiff problem
   whose character changes sharply along the integration. It is L-stable and its error is controlled
   with an embedded third order formula.

   Each step solves a three stage collocation system. Rather than factorizing the full 3*num_y
   system, the method uses the eigendecomposition of the Butcher matrix: one real eigenvalue and one
   complex conjugate pair. That turns each iteration into one real solve and one complex solve of
   size num_y, which is where `c_lu`'s complex routines come in.

   The implementation follows SciPy's `scipy.integrate.Radau`.

   References
   ----------
   .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II: Stiff and
          Differential-Algebraic Problems", Sec. IV.8.
*/

// Number of collocation stages.
const size_t RADAU_NUM_STAGES = 3;
// Maximum number of simplified Newton iterations allowed per attempt.
const size_t RADAU_NEWTON_MAX_ITER = 6;
// The error is controlled with a third order embedded formula, so the first step is sized with the
// exponent that an order 3 method would use.
const double RADAU_FIRST_STEP_ERROR_EXPONENT = 1.0 / (3.0 + 1.0);
// The interpolating polynomial is cubic, so the dense output carries three coefficient columns.
const size_t RADAU_INTERPOLATOR_POWER = 3;
// Above this convergence rate a step that needed several Newton iterations is taken as a sign that
// the Jacobian has gone stale.
const double RADAU_JACOBIAN_REFRESH_RATE = 1.0e-3;
// A step size change smaller than this is not worth the refactorization it would cost.
const double RADAU_MIN_STEP_CHANGE_FACTOR = 1.2;

class RADAU : public CySolverBase {

// Attributes
protected:
    // Collocation points and the embedded error estimator weights.
    double C[RADAU_NUM_STAGES]     = { 0.0 };
    double E[RADAU_NUM_STAGES]     = { 0.0 };
    // Transformation between the stage values Z and the transformed variables W.
    double T_matrix[RADAU_NUM_STAGES * RADAU_NUM_STAGES]  = { 0.0 };
    double TI_matrix[RADAU_NUM_STAGES * RADAU_NUM_STAGES] = { 0.0 };
    // Rows of TI, split into the real eigenvalue's row and the complex pair's row.
    double TI_real[RADAU_NUM_STAGES] = { 0.0 };
    std::complex<double> TI_complex[RADAU_NUM_STAGES] = { std::complex<double>(0.0, 0.0) };
    // Interpolator coefficients; P has shape (RADAU_NUM_STAGES, RADAU_INTERPOLATOR_POWER).
    double P_matrix[RADAU_NUM_STAGES * RADAU_INTERPOLATOR_POWER] = { 0.0 };
    // Eigenvalues of the inverse Butcher matrix.
    double mu_real                  = 0.0;
    std::complex<double> mu_complex = std::complex<double>(0.0, 0.0);

    // Iteration matrices and their factorizations.
    std::vector<double> jacobian_vec               = std::vector<double>(PRE_ALLOC_NUMY * PRE_ALLOC_NUMY);
    std::vector<double> lu_real_vec                = std::vector<double>(PRE_ALLOC_NUMY * PRE_ALLOC_NUMY);
    std::vector<std::complex<double>> lu_complex_vec = std::vector<std::complex<double>>(PRE_ALLOC_NUMY * PRE_ALLOC_NUMY);
    std::vector<int> pivot_real_vec                = std::vector<int>(PRE_ALLOC_NUMY);
    std::vector<int> pivot_complex_vec             = std::vector<int>(PRE_ALLOC_NUMY);
    double* jacobian_ptr                           = nullptr;
    double* lu_real_ptr                            = nullptr;
    std::complex<double>* lu_complex_ptr           = nullptr;
    int* pivot_real_ptr                            = nullptr;
    int* pivot_complex_ptr                         = nullptr;

    /* Per-step scratch space. Z holds the stage increments and W their transform; both are laid out
       stage-major, so stage i for variable y_j is at `[i * num_y + j]`. */
    std::vector<double> work_vec = std::vector<double>(PRE_ALLOC_NUMY * (4 * RADAU_NUM_STAGES + RADAU_INTERPOLATOR_POWER + 4));
    double* Z_ptr           = nullptr;  // Stage increments.
    double* Z0_ptr          = nullptr;  // Starting guess for the stage increments.
    double* W_ptr           = nullptr;  // Transformed stage increments.
    double* F_ptr           = nullptr;  // Derivative at each stage.
    double* scale_ptr       = nullptr;  // Tolerance-weighted scaling of y.
    double* f_start_ptr     = nullptr;  // Derivative at the start of the step.
    double* error_ptr       = nullptr;  // Local error estimate.
    double* newton_dW_ptr   = nullptr;  // Correction produced by a single Newton iteration.
    double* dense_Q_ptr     = nullptr;  // Interpolator built from the last accepted step.
    std::vector<std::complex<double>> complex_work_vec = std::vector<std::complex<double>>(PRE_ALLOC_NUMY * 2);
    std::complex<double>* complex_rhs_ptr = nullptr;

    /* The interpolator is an offset from the solution at the start of the step it was built over.
       By the time the next step needs it for its starting guess, `y_old_ptr` has already moved on,
       so keep a copy here. */
    std::vector<double> dense_y_old_vec = std::vector<double>(PRE_ALLOC_NUMY);
    double* dense_y_old_ptr             = nullptr;

    // Current step state
    double h_abs             = 0.0;
    double h_abs_previous    = 0.0;
    double error_norm_previous = 0.0;
    double newton_tol        = 0.0;
    bool has_previous_error  = false;
    bool lu_is_valid         = false;
    bool jacobian_is_current = false;

    // State of the interpolator built from the previous accepted step, used to warm start the
    // Newton iteration of the next one.
    bool has_dense_state = false;
    double dense_t_old   = 0.0;
    double dense_step    = 0.0;

// Methods
protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual CyrkErrorCodes p_finalize_setup() noexcept override;
    virtual void p_step_implementation() noexcept override;

    // Refresh `jacobian_ptr` from either the user's analytic Jacobian or finite differences.
    void p_update_jacobian() noexcept;
    // Factorize both iteration matrices for the given step size. Returns false if either is singular.
    bool p_factor_iteration_matrices(const double step) noexcept;
    // Run the simplified Newton iteration for one candidate step. Reports how many iterations were
    // used and the rate at which they were converging.
    bool p_solve_collocation_system(
        const double step,
        size_t* num_iterations_ptr,
        double* convergence_rate_ptr) noexcept;
    // Build the cubic interpolator from the stage increments of the accepted step.
    void p_build_dense_state(const double step) noexcept;
    // Predict the factor by which the step size should change.
    double p_predict_step_factor(const double h_abs_now, const double error_norm) const noexcept;
    // Root mean square norm of `vector_ptr / scale_ptr`, which is the norm used for error control.
    double p_scaled_norm(const double* vector_ptr, const double* scale_ptr_) const noexcept;

public:
    // Copy over base class constructors
    using CySolverBase::CySolverBase;

    virtual void set_Q_order(size_t* Q_order_ptr) override;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
};
