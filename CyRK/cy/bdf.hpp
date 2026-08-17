#pragma once

#include <vector>

#include "c_common.hpp"
#include "cysolver.hpp"

// ####################################################################################################################
// BDF Integrator
// ####################################################################################################################
/* Implicit multi-step integrator based on the backward differentiation formulas.

   The order varies automatically between 1 and 5 and a quasi-constant step size is used, which
   means that the solution history is rescaled whenever the step size changes rather than being
   rebuilt. The implementation follows SciPy's `scipy.integrate.BDF`, including the accuracy
   enhancement from the modified formulas (NDF).

   Each step solves a non-linear algebraic system with a simplified Newton iteration. The
   iteration matrix `I - c * J` is factorized with CyRK's own LU routines and is reused across
   steps until the step size, the order, or a convergence failure forces a refactorization.

   References
   ----------
   .. [1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical Solution of Ordinary
          Differential Equations", ACM Transactions on Mathematical Software, Vol. 1, No. 1,
          pp. 71-96, March 1975.
   .. [2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI. COMPUTE., Vol. 18,
          No. 1, pp. 1-22, January 1997.
   .. [3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems",
          Sec. III.2.
*/

// Maximum number of simplified Newton iterations allowed per attempt.
const size_t BDF_NEWTON_MAX_ITER = 4;
// The error of a BDF step of order k behaves like step ** (k + 1), so the first step is sized
// with the same exponent that an order 1 method would use.
const double BDF_FIRST_STEP_ERROR_EXPONENT = 1.0 / (1.0 + 1.0);

class BDF : public CySolverBase {

// Attributes
protected:
    // Method coefficients. `gamma` and `alpha` are indexed by order; `error_const` needs one more
    // entry so that the order-increase error estimate can be formed at the maximum order.
    double gamma[BDF_MAX_ORDER + 1]       = { 0.0 };
    double alpha[BDF_MAX_ORDER + 1]       = { 0.0 };
    double error_const[BDF_MAX_ORDER + 2] = { 0.0 };

    /* Backward difference history. Row j starts at `&D_ptr[j * num_y]`. Rows 0 through `order`
       hold the differences that define the current interpolating polynomial; the two extra rows
       are scratch used by the error estimates that decide whether to change order. */
    std::vector<double> D_vec = std::vector<double>((BDF_MAX_ORDER + 3) * PRE_ALLOC_NUMY);
    double* D_ptr             = nullptr;

    // Iteration matrix and its LU factorization (both stored column-major).
    std::vector<double> jacobian_vec = std::vector<double>(PRE_ALLOC_NUMY * PRE_ALLOC_NUMY);
    std::vector<double> lu_vec       = std::vector<double>(PRE_ALLOC_NUMY * PRE_ALLOC_NUMY);
    std::vector<int> pivot_vec       = std::vector<int>(PRE_ALLOC_NUMY);
    double* jacobian_ptr             = nullptr;
    double* lu_ptr                   = nullptr;
    int* pivot_ptr                   = nullptr;

    /* Per-step scratch space. All of these are sub-arrays of `work_vec` so that the solver only
       performs a single allocation. */
    std::vector<double> work_vec = std::vector<double>(PRE_ALLOC_NUMY * (5 + BDF_MAX_ORDER + 1));
    double* y_predict_ptr = nullptr;  // Explicit prediction of y at the end of the step.
    double* psi_ptr       = nullptr;  // History term of the algebraic system.
    double* scale_ptr     = nullptr;  // Tolerance-weighted scaling of y.
    double* d_total_ptr   = nullptr;  // Total Newton correction; also the local error estimate.
    double* newton_dy_ptr = nullptr;  // Correction produced by a single Newton iteration.
    double* D_temp_ptr    = nullptr;  // Holds the rescaled history while `D` is being changed.

    // Current step state
    size_t order            = 1;
    size_t num_equal_steps  = 0;
    double h_abs            = 0.0;
    double newton_tol       = 0.0;
    double c_of_last_factor = 0.0;
    bool lu_is_valid        = false;
    bool jacobian_is_current = false;

// Methods
protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual CyrkErrorCodes p_finalize_setup() noexcept override;
    virtual void p_step_implementation() noexcept override;

    // Rescale the difference array so that it describes the same polynomial sampled at a new step
    // size, where `factor` is the ratio of the new step size to the old one.
    void p_change_D(const size_t order_, const double factor) noexcept;
    // Refresh `jacobian_ptr` from either the user's analytic Jacobian or finite differences.
    void p_update_jacobian() noexcept;
    // Factorize `I - c * J`. Returns false if the matrix is singular.
    bool p_factor_iteration_matrix(const double c) noexcept;
    // Run the simplified Newton iteration for one candidate step. Returns true on convergence and
    // reports how many iterations were used.
    bool p_solve_newton_system(const double t_new, const double c, size_t* num_iterations_ptr) noexcept;
    // Root mean square norm of `vector_ptr / scale_ptr`, which is the norm used for error control.
    double p_scaled_norm(const double* vector_ptr, const double* scale_ptr_) const noexcept;

public:
    // Copy over base class constructors
    using CySolverBase::CySolverBase;

    virtual void set_Q_order(size_t* Q_order_ptr) override;
    virtual void set_Q_order_max(size_t* Q_order_max_ptr) override;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
    virtual double get_dense_step() const noexcept override;
    virtual double* get_dense_base_y_ptr() noexcept override;
};
