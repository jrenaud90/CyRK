#pragma once

#include <cmath>
#include <cstring>
#include <cstdio>

#include <memory>

#include "cysolution.hpp"
#include "common.hpp"

class CySolverBase {


// Attributes
protected:
    // ** Attributes **
    // Status variables
    bool reset_called = false;

    // Time variables
    double t_start = 0.0;
    double t_end = 0.0;
    double t_old = 0.0;
    double t_delta = 0.0;
    double t_delta_abs = 0.0;
    double direction_inf = 0.0;
    bool direction_flag = false;

    // Dependent variables
    double num_y_dbl = 0.0;
    double num_y_sqrt = 0.0;
    size_t num_dy = 0;
    // The size of the stack allocated tracking arrays is equal to the maximum allowed `num_y` (25).
    double y0[25] = { 0.0 };
    double y_old[25] = { 0.0 };
    double y_now[25] = { 0.0 };
    // For dy, both the dy/dt and any extra outputs are stored. So the maximum size is `num_y` (25) + `num_extra` (25)
    double dy_old[50] = { 0.0 };
    double dy_now[50] = { 0.0 };

    // dy_now_ptr and y_now_ptr are declared in public.
    double* y_old_ptr = &y_old[0];
    double* dy_old_ptr = &dy_old[0];

    // Integration step information
    size_t len_t = 0;
    size_t max_num_steps = 0;
    bool user_provided_max_num_steps = false;

    // Information on capturing extra information during integration.
    bool capture_extra = false;
    size_t num_extra = 0;

    // Differential equation information
    double* args_ptr = nullptr;
    DiffeqFuncType diffeq_ptr = nullptr;

public:

    // Result storage
    std::shared_ptr<CySolverResult> storage_ptr = nullptr;

    // Status attributes
    int status = -999;

    // State attributes
    size_t num_y = 0;
    double t_now;
    double* y0_ptr = &y0[0];
    double* y_now_ptr = &y_now[0];
    double* dy_now_ptr = &dy_now[0];


// Methods
protected:
    virtual void p_step_implementation();

public:
    CySolverBase();
    virtual ~CySolverBase();
    CySolverBase(
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
        size_t max_ram_MB = 2000
    );
    
    bool check_status();
    void diffeq();
    void take_step();
    void change_storage(std::shared_ptr<CySolverResult> new_storage_ptr, bool auto_reset = true);
    virtual void reset();
};