#pragma once

#include <cstring>

#include <algorithm>
#include <functional>
#include <memory>

#include "common.hpp"
#include "cysolution.hpp"
#include "dense.hpp"
#include "cy_array.hpp"

// !!!
// Comment the following 
// typedef DiffeqFuncType DiffeqMethod;
// Comment this import if working outside of CyRK and you just want the program to compile and run for testing/developing the C++ only code.
// "pysolver_cyhook_api.h" is generated by Cython when building the CyRK project.
// It is based off of the "pysolver_cyhook.pyx" file. 
// Read more about how C++ can call python functions here:
// https://stackoverflow.com/questions/10126668/can-i-override-a-c-virtual-function-within-python-with-cython
// and here: https://github.com/dashesy/pyavfcam/blob/master/src/avf.pyx#L27
#include <Python.h>
#include "pysolver_cyhook_api.h"

struct _object;
typedef _object PyObject;

class CySolverBase {

// Methods
protected:
    virtual void p_estimate_error();
    virtual void p_step_implementation();
    virtual CySolverDense* p_dense_output_heap();
    virtual void p_dense_output_stack(CySolverDense& dense_output_ptr);

public:
    CySolverBase();
    virtual ~CySolverBase();
    CySolverBase(
        // Input variables
        DiffeqFuncType diffeq_ptr,
        std::shared_ptr<CySolverResult> const storage_ptr,
        const double t_start,
        const double t_end,
        const double* const y0_ptr,
        const unsigned int num_y,
        const unsigned int num_extra = 0,
        const void* const args_ptr = nullptr,
        const size_t max_num_steps = 0,
        const size_t max_ram_MB = 2000,
        const bool use_dense_output = false,
        const double* t_eval = nullptr,
        const size_t len_t_eval = 0
    );

    void change_storage(std::shared_ptr<CySolverResult> new_storage_ptr, bool auto_reset = true);
    bool check_status() const;
    virtual void reset();
    virtual void calc_first_step_size();
    void take_step();
    void solve();
    // Diffeq can either be the C++ class method or the python hook diffeq. By default set to C++ version.
    void cy_diffeq() noexcept;
    std::function<void(CySolverBase*)> diffeq;

    // PySolver methods
    void set_cython_extension_instance(PyObject* cython_extension_class_instance, DiffeqMethod py_diffeq_method);
    void py_diffeq();

// Attributes
protected:
    // ** Attributes **

    // Time variables
    double t_start       = 0.0;
    double t_end         = 0.0;
    double t_old         = 0.0;
    double t_delta       = 0.0;
    double t_delta_abs   = 0.0;
    double direction_inf = 0.0;

    // Dependent variables
    double num_y_dbl  = 0.0;
    double num_y_sqrt = 0.0;
    // The size of the stack allocated tracking arrays is equal to the maximum allowed `num_y` (25).
    double y0[Y_LIMIT]    = { 0.0 };
    double y_old[Y_LIMIT] = { 0.0 };
    double y_now[Y_LIMIT] = { 0.0 };
    // For dy, both the dy/dt and any extra outputs are stored. So the maximum size is `num_y` (25) + `num_extra` (25)
    double dy_old[DY_LIMIT] = { 0.0 };
    double dy_now[DY_LIMIT] = { 0.0 };

    // dy_now_ptr and y_now_ptr are declared in public.
    double* const y0_ptr     = &y0[0];
    double* const y_old_ptr  = &y_old[0];
    double* const dy_old_ptr = &dy_old[0];

    // Integration step information
    size_t max_num_steps = 0;

    // Differential equation information
    const void* args_ptr    = nullptr;
    DiffeqFuncType diffeq_ptr = nullptr;
    
    // t_eval information
    std::vector<double> t_eval_vec = std::vector<double>(0);
    double* t_eval_ptr       = t_eval_vec.data();
    size_t t_eval_index_old  = 0;
    size_t len_t_eval        = 0;
    bool use_t_eval          = false;

    // Information on capturing extra information during integration.
    int num_extra = 0;

    // Keep bools together to reduce size
    bool direction_flag = false;
    bool reset_called   = false;
    bool capture_extra  = false;
    bool user_provided_max_num_steps = false;

    // Dense (Interpolation) Attributes
    bool use_dense_output = false;

public:
    // PySolver Attributes
    bool use_pysolver = false;
    DiffeqMethod py_diffeq_method = nullptr;
    PyObject* cython_extension_class_instance = nullptr;

    // Status attributes
    int status = -999;
    int integration_method = -1;

    // Meta data
    unsigned int num_dy = 0;
    unsigned int num_y  = 0;

    // Result storage
    std::shared_ptr<CySolverResult> storage_ptr = std::make_shared<CySolverResult>();

    // State attributes
    size_t len_t       = 0;
    double t_now[1]    = {0.0};
    double* const t_now_ptr  = &t_now[0];
    double* const y_now_ptr  = &y_now[0];
    double* const dy_now_ptr = &dy_now[0];
};
