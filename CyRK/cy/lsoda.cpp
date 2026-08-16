#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "lsoda.hpp"
#include "dense.hpp"
#include "cysolution.hpp"

// ########################################################################################################################
// LSODAConfig
// ########################################################################################################################
bool LSODAConfig::banded_jacobian() const noexcept
{
    return (this->num_lower != MAX_SIZET_SIZE) or (this->num_upper != MAX_SIZET_SIZE);
}

void LSODAConfig::initialize()
{
    ProblemConfig::initialize();
    this->initialized = false;

    if ((this->max_order_nonstiff == 0) or (this->max_order_stiff == 0))
    {
        throw std::length_error("Unexpected LSODA maximum order; must be at least 1.");
    }
    if (this->banded_jacobian())
    {
        // A partially specified band is treated as zero width on the unspecified side.
        const size_t l_num_lower = (this->num_lower == MAX_SIZET_SIZE) ? 0 : this->num_lower;
        const size_t l_num_upper = (this->num_upper == MAX_SIZET_SIZE) ? 0 : this->num_upper;
        if ((l_num_lower >= this->num_y) or (l_num_upper >= this->num_y))
        {
            throw std::length_error("Unexpected LSODA Jacobian bandwidth; must be less than the number of dependent variables.");
        }
    }

    this->initialized = true;
}

void LSODAConfig::update_properties_from_config(ProblemConfig* new_config_ptr)
{
    LSODAConfig* const lsoda_config_ptr = dynamic_cast<LSODAConfig*>(new_config_ptr);
    if (lsoda_config_ptr)
    {
        this->min_step_size      = lsoda_config_ptr->min_step_size;
        this->max_order_nonstiff = lsoda_config_ptr->max_order_nonstiff;
        this->max_order_stiff    = lsoda_config_ptr->max_order_stiff;
        this->num_lower          = lsoda_config_ptr->num_lower;
        this->num_upper          = lsoda_config_ptr->num_upper;
    }

    ProblemConfig::update_properties_from_config(new_config_ptr);
}

// #####################################################################################################################
// LSODA Integrator
// #####################################################################################################################
/* ========================================================================= */
/* =========================  Protected Methods  =========================== */
/* ========================================================================= */
void LSODA::p_diffeq_hook(int* neq, double* t, double* y, double* ydot, void* user_data)
{
    (void)neq;
    LSODA* const solver_ptr = static_cast<LSODA*>(user_data);

    solver_ptr->t_now = t[0];
    if (y != solver_ptr->y_now_ptr)
    {
        std::memcpy(solver_ptr->y_now_ptr, y, solver_ptr->sizeof_dbl_Ny);
    }
    solver_ptr->diffeq(solver_ptr);
    std::memcpy(ydot, solver_ptr->dy_now_ptr, solver_ptr->sizeof_dbl_Ny);
}

void LSODA::p_jacobian_hook(int* neq, double* t, double* y, int* ml, int* mu, double* pd, int* nrowpd, void* user_data)
{
    // The band structure and leading dimension are already known to the configured Jacobian.
    (void)neq;
    (void)ml;
    (void)mu;
    (void)nrowpd;
    LSODA* const solver_ptr = static_cast<LSODA*>(user_data);

    solver_ptr->t_now = t[0];
    if (y != solver_ptr->y_now_ptr)
    {
        std::memcpy(solver_ptr->y_now_ptr, y, solver_ptr->sizeof_dbl_Ny);
    }
    solver_ptr->p_call_jacobian(pd);
}

CyrkErrorCodes LSODA::p_convert_istate(const int istate_) noexcept
{
    switch (istate_)
    {
    case -1:
        // Too many internal steps were needed to reach the next output point.
        return CyrkErrorCodes::MAX_ITERATIONS_HIT;
    case -2:
        // The requested accuracy is finer than the machine can deliver at the current t.
        return CyrkErrorCodes::STEP_SIZE_ERROR_ACCEPTANCE;
    case -4:
        // The error test failed repeatedly at the smallest allowed step size.
        return CyrkErrorCodes::STEP_SIZE_ERROR_SPACING;
    case -5:
        // The corrector iteration failed to converge repeatedly.
        return CyrkErrorCodes::NEWTON_CONVERGENCE_ERROR;
    case -6:
        // A solution component vanished while its absolute tolerance was zero.
        return CyrkErrorCodes::BAD_CONFIG_DATA;
    case -7:
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    [[unlikely]] default:
        return CyrkErrorCodes::LSODA_INTERNAL_ERROR;
    }
}

CyrkErrorCodes LSODA::p_additional_setup() noexcept
{
    this->integration_method = ODEMethod::LSODA;

    LSODAConfig* const config_ptr = dynamic_cast<LSODAConfig*>(this->storage_ptr->config_uptr.get());
    if (not config_ptr) [[unlikely]]
    {
        // LSODA needs its own configuration class; `CySolverResult` builds one alongside the solver.
        return CyrkErrorCodes::BAD_CONFIG_DATA;
    }

    // Decide whether the Jacobian is dense or banded and pick the matching ODEPACK "jt" code.
    this->use_banded_jacobian = config_ptr->banded_jacobian();
    this->num_lower = (config_ptr->num_lower == MAX_SIZET_SIZE) ? 0 : config_ptr->num_lower;
    this->num_upper = (config_ptr->num_upper == MAX_SIZET_SIZE) ? 0 : config_ptr->num_upper;
    if (this->use_banded_jacobian)
    {
        this->jacobian_type = this->jac_ptr ? 4 : 5;
    }
    else
    {
        this->jacobian_type = this->jac_ptr ? 1 : 2;
    }

    // Size the ODEPACK work arrays.
    const size_t rwork_size = this->use_banded_jacobian
        ? c_lsoda_banded_rwork_size(this->num_y, this->num_lower, this->num_upper)
        : c_lsoda_dense_rwork_size(this->num_y);
    const size_t iwork_size = c_lsoda_iwork_size(this->num_y);
    if ((rwork_size > MAX_INT_SIZE) or (iwork_size > MAX_INT_SIZE)) [[unlikely]]
    {
        // ODEPACK addresses its work arrays with ints.
        return CyrkErrorCodes::VECTOR_SIZE_EXCEEDS_LIMITS;
    }
    /* With a dense Jacobian the work array grows with the square of the number of dependent
       variables. Check it against the user's RAM budget; narrowing the Jacobian to a band brings
       this back down to a linear cost. */
    const double rwork_MB = (double)rwork_size * (double)sizeof(double) / (1024.0 * 1024.0);
    if (rwork_MB > (double)config_ptr->max_ram_MB) [[unlikely]]
    {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }
    try
    {
        this->rwork_vec.resize(rwork_size);
        this->iwork_vec.resize(iwork_size);
    }
    catch (const std::bad_alloc&)
    {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }
    std::fill(this->rwork_vec.begin(), this->rwork_vec.end(), 0.0);
    std::fill(this->iwork_vec.begin(), this->iwork_vec.end(), 0);
    this->rwork_ptr = this->rwork_vec.data();
    this->iwork_ptr = this->iwork_vec.data();

    // Clear the common block so that a reused solver does not inherit any previous run's state.
    this->common_state = c_lsoda_common_t();
    this->istate       = 1;

    return CyrkErrorCodes::NO_ERROR;
}

void LSODA::p_calc_first_step_size() noexcept
{
    // ODEPACK derives its own first step size from the initial derivative and the tolerances.
    this->step_size = 0.0;
}

CyrkErrorCodes LSODA::p_finalize_setup() noexcept
{
    LSODAConfig* const config_ptr = dynamic_cast<LSODAConfig*>(this->storage_ptr->config_uptr.get());
    if (not config_ptr) [[unlikely]]
    {
        return CyrkErrorCodes::BAD_CONFIG_DATA;
    }

    // Which of the tolerances were provided as arrays (ODEPACK's "itol").
    if (this->use_array_rtols)
    {
        this->itol = this->use_array_atols ? 4 : 3;
    }
    else
    {
        this->itol = this->use_array_atols ? 2 : 1;
    }

    // Optional inputs. `iopt = 1` in the step call tells ODEPACK to read all of these.
    if (this->use_banded_jacobian)
    {
        this->iwork_ptr[0] = (int)this->num_lower;
        this->iwork_ptr[1] = (int)this->num_upper;
    }
    this->iwork_ptr[4] = 0;                  // ixpr: do not print method switches.
    // CyRK enforces its own step limit through `max_num_steps`, so ODEPACK's per-call limit is set
    // as high as it can go. Only one step is taken per call anyway.
    this->iwork_ptr[5] = (int)MAX_INT_SIZE;  // mxstep
    this->iwork_ptr[6] = 0;                  // mxhnil: use the default warning limit.
    this->iwork_ptr[7] = (int)std::min<size_t>(config_ptr->max_order_nonstiff, LSODA_MAX_ORDER);
    this->iwork_ptr[8] = (int)std::min<size_t>(config_ptr->max_order_stiff, BDF_MAX_ORDER);

    // The user's first step size is unsigned but ODEPACK wants it signed.
    this->rwork_ptr[4] = this->user_provided_first_step_size * (this->direction_flag ? 1.0 : -1.0);
    // ODEPACK uses zero rather than infinity to mean "no maximum step size".
    this->rwork_ptr[5] = std::isinf(this->max_step_size) ? 0.0 : this->max_step_size;
    this->rwork_ptr[6] = config_ptr->min_step_size;

    return CyrkErrorCodes::NO_ERROR;
}

void LSODA::p_step_implementation() noexcept
{
    // ODEPACK's itask 5 takes a single step without stepping past the critical time in rwork[0].
    int itask     = 5;
    int iopt      = 1;
    double t_used = this->t_old;
    double t_out  = this->t_end;
    this->rwork_ptr[0] = this->t_end;

    c_lsoda(
        &LSODA::p_diffeq_hook,
        (int)this->num_y,
        this->y_now_ptr,
        &t_used,
        &t_out,
        this->itol,
        this->rtols_ptr,
        this->atols_ptr,
        &itask,
        &this->istate,
        &iopt,
        this->rwork_ptr,
        (int)this->rwork_vec.size(),
        this->iwork_ptr,
        (int)this->iwork_vec.size(),
        &LSODA::p_jacobian_hook,
        this->jacobian_type,
        &this->common_state,
        this);

    if (this->istate < 0) [[unlikely]]
    {
        this->error_flag = true;
        this->storage_ptr->update_status(LSODA::p_convert_istate(this->istate));
        return;
    }

    this->t_now = t_used;

    // Extra outputs are not tracked by ODEPACK, so refresh them at the accepted state.
    if (this->capture_extra)
    {
        this->diffeq(this);
    }
}

/* ========================================================================= */
/* =========================  Public Methods  ============================== */
/* ========================================================================= */
/* Dense Output Methods */
void LSODA::set_Q_order(size_t* Q_order_ptr)
{
    /* iwork[13] holds the order that the last successful step actually used. The Nordsieck array
       has that many columns beyond the first, and the first column is stored separately as the
       interpolant's base value. */
    const int order_used = this->iwork_ptr ? this->iwork_ptr[13] : 0;
    Q_order_ptr[0] = (order_used > 0) ? (size_t)order_used : 1;
}

void LSODA::set_Q_order_max(size_t* Q_order_max_ptr)
{
    Q_order_max_ptr[0] = LSODA_MAX_ORDER;
}

void LSODA::set_Q_array(double* Q_ptr) noexcept
{
    /* The Nordsieck history lives at rwork[20] onward in column-major order with `num_y` rows.
       Column j holds the j-th scaled derivative of the interpolating polynomial about `t_now`. */
    const size_t l_num_y = this->num_y;
    size_t Q_order = 0;
    this->set_Q_order(&Q_order);

    const double* const nordsieck_ptr = &this->rwork_ptr[20];
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        const size_t stride_Q = y_i * Q_order;
        for (size_t column_i = 1; column_i <= Q_order; column_i++)
        {
            Q_ptr[stride_Q + column_i - 1] = nordsieck_ptr[y_i + column_i * l_num_y];
        }
    }

    /* iwork[14] is the order that will be attempted next. When the order is about to drop,
       ODEPACK leaves the final column scaled to the previous step size because it will not be
       used again, so rescale the copy here (rwork[10] is the step size that was actually used). */
    const size_t next_order = (size_t)std::max<int>(this->iwork_ptr[14], 0);
    if (next_order < Q_order)
    {
        const double step_used = this->rwork_ptr[10];
        if (step_used != 0.0) [[likely]]
        {
            const double column_scale = std::pow(this->rwork_ptr[11] / step_used, (double)Q_order);
            for (size_t y_i = 0; y_i < l_num_y; y_i++)
            {
                Q_ptr[y_i * Q_order + Q_order - 1] *= column_scale;
            }
        }
    }
}

double LSODA::get_dense_step() const noexcept
{
    // rwork[11] is the step size that the Nordsieck array is currently scaled to.
    return this->rwork_ptr ? this->rwork_ptr[11] : 0.0;
}

double* LSODA::get_dense_base_y_ptr() noexcept
{
    // The first column of the Nordsieck array is the solution at the end of the step.
    return &this->rwork_ptr[20];
}
