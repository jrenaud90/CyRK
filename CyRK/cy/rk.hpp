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


// #####################################################################################################################
// Tsitouras 5(4)
// #####################################################################################################################
/* Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption",
   Computers & Mathematics with Applications 62 (2011) 770-775. Seven stages, the last of which is the derivative at the
   end of the step (first same as last). Its fourth-order interpolant uses no extra stages. */
const size_t Tsit5_order     = 5;
const size_t Tsit5_n_stages  = 6;
const size_t Tsit5_len_Pcols = 4;
const size_t Tsit5_error_estimator_order = 4;
const double Tsit5_error_exponent = 1.0 / (4.0 + 1.0);  // Defined as 1 / (error_order + 1)
// Interpolation nodes theta_j (Chebyshev-Lobatto points rounded to multiples of 1/1024) and the barycentric
// weights of the dense output, which stores the interpolant's values at these nodes ("dense.cpp").
const double Tsit5_dense_nodes[Tsit5_len_Pcols] = {
    (75.0 / 512.0),
    (1.0 / 2.0),
    (437.0 / 512.0),
    1.0,
};
const double Tsit5_dense_weights[Tsit5_len_Pcols] = {
    (-34359738368.0 / 1073741775.0),
    (1048576.0 / 32761.0),
    (-34359738368.0 / 1073741775.0),
    (524288.0 / 32775.0),
};

class Tsit5 : public RKSolver {

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
// Verner 7(6) "most efficient"
// #####################################################################################################################
/* J. H. Verner, "Numerically optimal Runge-Kutta pairs with interpolants", Numerical Algorithms 53 (2010) 383-396.
   Ten stages: the tenth is used only by the error estimate. The seventh-order interpolant uses the derivative at the
   end of the step and five extra stages, which are evaluated only when an interpolant is needed. */
const size_t Vern7_order          = 7;
const size_t Vern7_n_stages       = 10;
const size_t Vern7_nEXTRA_stages  = 5;
const size_t Vern7_len_Pcols      = 7;
const size_t Vern7_error_estimator_order = 6;
const double Vern7_error_exponent = 1.0 / (6.0 + 1.0);  // Defined as 1 / (error_order + 1)
// Interpolation nodes theta_j (Chebyshev-Lobatto points rounded to multiples of 1/1024) and the barycentric
// weights of the dense output, which stores the interpolant's values at these nodes ("dense.cpp").
const double Vern7_dense_nodes[Vern7_len_Pcols] = {
    (51.0 / 1024.0),
    (193.0 / 1024.0),
    (199.0 / 512.0),
    (313.0 / 512.0),
    (831.0 / 1024.0),
    (973.0 / 1024.0),
    1.0,
};
const double Vern7_dense_weights[Vern7_len_Pcols] = {
    1.16763066592238045285302918553247884e3,
    -1.1735348210477018068472360070981182e3,
    1.17346020808882593994157618084401926e3,
    -1.17346020808882593994157618084401926e3,
    1.1735348210477018068472360070981182e3,
    -1.16763066592238045285302918553247884e3,
    5.81434751558841151434214556410920378e2,
};

class Vern7 : public RKSolver {

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
// Verner 8(7) "most efficient"
// #####################################################################################################################
/* J. H. Verner (2010), as for Vern7. Thirteen stages: the thirteenth is used only by the error estimate. The
   eighth-order interpolant uses the derivative at the end of the step and seven extra stages, which are evaluated
   only when an interpolant is needed. */
const size_t Vern8_order          = 8;
const size_t Vern8_n_stages       = 13;
const size_t Vern8_nEXTRA_stages  = 7;
const size_t Vern8_len_Pcols      = 8;
const size_t Vern8_error_estimator_order = 7;
const double Vern8_error_exponent = 1.0 / (7.0 + 1.0);  // Defined as 1 / (error_order + 1)
// Interpolation nodes theta_j (Chebyshev-Lobatto points rounded to multiples of 1/1024) and the barycentric
// weights of the dense output, which stores the interpolant's values at these nodes ("dense.cpp").
const double Vern8_dense_nodes[Vern8_len_Pcols] = {
    (39.0 / 1024.0),
    (75.0 / 512.0),
    (79.0 / 256.0),
    (1.0 / 2.0),
    (177.0 / 256.0),
    (437.0 / 512.0),
    (985.0 / 1024.0),
    1.0,
};
const double Vern8_dense_weights[Vern8_len_Pcols] = {
    -4.09478588289711542667625502294735351e3,
    4.09824656413001727960397450016225089e3,
    -4.09721732659644807580462449965386734e3,
    4.09456873457960173693316886744546452e3,
    -4.09721732659644807580462449965386734e3,
    4.09824656413001727960397450016225089e3,
    -4.09478588289711542667625502294735351e3,
    2.0464722780737453544103205887162377e3,
};

class Vern8 : public RKSolver {

protected:
    virtual CyrkErrorCodes p_additional_setup() noexcept override;
    virtual void p_compute_stages() noexcept override;
    virtual double p_estimate_error() noexcept override;
public:
    // Copy over base class constructors
    using RKSolver::RKSolver;
    virtual void set_Q_array(double* Q_ptr) noexcept override;
};

// Storage that the dense output needs to evaluate the node-value interpolants above ("dense.cpp").
const size_t RK_MAX_DENSE_NODES = 8;
static_assert(
    (Tsit5_len_Pcols <= RK_MAX_DENSE_NODES) && (Vern7_len_Pcols <= RK_MAX_DENSE_NODES) &&
    (Vern8_len_Pcols <= RK_MAX_DENSE_NODES),
    "RK_MAX_DENSE_NODES must cover every node-value interpolant.");

