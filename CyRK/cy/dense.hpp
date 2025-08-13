#pragma once

#include <memory>
#include <functional>
#include <vector>
#include <cstring>

#include "common.hpp"
#include "cysolver.hpp"


class CySolverDense {
    /* Many of these dense classes will be built if the integration is long and `dense_output=True`. It needs to be as light weight as possible.
    Keep in mind that its true size equals the below structure + the size of the state_vector which is heap allocated and size = num_y * Q_order.
    */
    /* Attributes */

protected:
    // y and t state info
    // Dense state variables
    size_t Q_order = 0;
    size_t num_y   = 0;  // Number of dependent variables

    // Pointer to the CySolverBase class
    CySolverBase* solver_ptr = nullptr;

    // Time step info
    double t_old = 0.0;
    double t_now = 0.0;
    double step  = 0.0;

    // Vectors for stored data (stored at each step)
    // Q has shape of (num_y, q_order + 1)
    // The max size of Q is (7) * num_y for DOP853; set default for RK45 of 4
    // +1 is so we can store y_values in the first spot.
    std::vector<double> state_data_vec = std::vector<double>(PRE_ALLOC_NUMY * (4 + 1));

public:

/* Methods */
protected:

public:
    virtual ~CySolverDense();
    CySolverDense() {};
    CySolverDense(
        CySolverBase* solver_ptr_,
        bool set_state
        );

    virtual void set_state();
    virtual void call(double t_interp, double* y_interp_ptr);
};
