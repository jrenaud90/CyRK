#pragma once

/* This class is not subclasses because we want to be able to stack allocate it. It is built in a tight loop
* S o by avoiding calls to New we greatly improve performance.
*/

#include <memory>
#include <functional>
#include <vector>
#include <cstring>

#include "common.hpp"

class CySolverBase;


class CySolverDense
/* Many of these dense classes will be built if the integration is long and `dense_output=True`. It needs to be as light weight as possible. 
Keep in mind that its true size equals the below structure + the size of the state_vector which is heap allocated and size = num_y * Q_order.
*/
{
/* Attributes */
protected:

public:

    // Integrator info
    int integrator_int = -1;

    // y and t state info
    // Dense state variables
    size_t Q_order = 0;
    size_t num_y   = 0;

    // Pointer to the CySolverBase class
    CySolverBase* solver_ptr = nullptr;

    // Time step info
    double t_old = 0.0;
    double t_now = 0.0;
    double step  = 0.0;

    // Vectors for stored data (stored at each step)
    std::vector<double> state_data_vec;

/* Methods */
protected:

public:
    virtual ~CySolverDense();
    CySolverDense() {};
    CySolverDense(
        int integrator_int,
        CySolverBase* solver_ptr,
        bool set_state
        );

    virtual void set_state();
    virtual void call(double t_interp, double* y_interp_ptr);
};
