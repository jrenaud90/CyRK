#pragma once

/* This class is not subclasses because we want to be able to stack allocate it. It is built in a tight loop
* S o by avoiding calls to New we greatly improve performance.
*/

#include <memory>
#include <functional>
#include <cstring>

#include "common.hpp"

class CySolverBase;


class CySolverDense
{
/* Attributes */
protected:

    double y_stored[Y_LIMIT] = { };

public:

    // Integrator info
    int integrator_int = -1;

    // y and t state info
    // Dense state variables
    unsigned int Q_order = 0;
    unsigned int num_y     = 0;
    unsigned int num_extra = 0;

    // Pointer to the CySolverBase class
    CySolverBase* solver_ptr = nullptr;

    // Time step info
    double t_old = 0.0;
    double t_now = 0.0;
    double step  = 0.0;

    // Q is defined by Q = K.T.dot(self.P)  K has shape of (n_stages + 1, num_y) so K.T has shape of (num_y, n_stages + 1)
    // P has shape of (4, 3) for RK23; (7, 4) for RK45.. So (n_stages + 1, Q_order)
    // So Q has shape of (num_y, q_order)
    // The max size of Q is (7) * num_y. Lets assume this is the max size and stack allocate Q.
    double Q[Y_LIMIT * 7] = { };

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
