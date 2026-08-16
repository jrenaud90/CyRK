#include "dense.hpp"
#include "cysolver.hpp"

// Constructors
CySolverDense::CySolverDense(
            CySolverResult* solution_ptr_,
            bool set_state) :
        solution_ptr(solution_ptr_)
{
    this->setup(set_state);
}

// Destructors
CySolverDense::~CySolverDense()
{
}

void CySolverDense::setup(bool set_state)
{
    CySolverBase* solver_ptr = this->solution_ptr->solver_uptr.get();

    if (solver_ptr) [[likely]]
    {
        // Allocate memory for state vectors (memory allocation should not change since num_y does not change)
        solver_ptr->set_Q_order(&this->Q_order);
        solver_ptr->set_Q_order_max(&this->Q_order_max);

        // Resize state vector based on dimensions. The state vector is a combination of the current y-values and Q
        // Q is a matrix of solver-specific parameters at the current time.
        // Q is defined by Q = K.T.dot(self.P)  K has shape of (n_stages + 1, num_y) so K.T has shape of (num_y, n_stages + 1)
        // P has shape of (4, 3) for RK23; (7, 4) for RK45.. So (n_stages + 1, Q_order)
        // So Q has shape of (num_y, q_order)
        // The max size of Q is (7) * num_y for DOP853
        // state vector is laid out as [y_vector, Q_matrix]
        this->num_y  = solver_ptr->num_y;
        this->num_dy = solver_ptr->num_dy;
        this->state_data_vec.resize(this->num_y * (this->Q_order_max + 1));  // +1 is so we can store y_values in the first spot.
        this->initialized = true;

        // Populate values with current state
        if (set_state)
        {
            this->set_state();
        }
    }
}

void CySolverDense::set_state()
{
    if (this->initialized) [[likely]]
    {
        CySolverBase* solver_ptr = this->solution_ptr->solver_uptr.get();

        // Store time information
        this->t_old = solver_ptr->t_old;
        this->t_now = solver_ptr->t_now;

        // Step that the interpolant's Q array is scaled against. For the single-step methods this
        // is just the step that was taken; the multi-step methods scale their history to the step
        // that they intend to take next.
        this->step = solver_ptr->get_dense_step();

        // The order can change between steps for the multi-step methods, so refresh it here.
        solver_ptr->set_Q_order(&this->Q_order);

        // Make a copy of the y_in pointer in the state vector storage
        std::memcpy(this->state_data_vec.data(), solver_ptr->get_dense_base_y_ptr(), sizeof(double) * this->num_y);

        // Tell the solver to populate the values of the Q matrix. 
        // Q starts at the num_y location of the state vector
        solver_ptr->set_Q_array(&this->state_data_vec[this->num_y]);

        this->state_set = true;
    }
}

void CySolverDense::call(double t_interp, double* y_interp_ptr)
{
    double step_factor = (t_interp - this->t_old) / this->step;

    // SciPy Step:: p = np.tile(x, self.order + 1) (scipy order is Q_order - 1)
    // Create pointers to the y and Q sub components of the state vector for ease of reading
    double* y_stored_ptr = &this->state_data_vec[0];
    double* Q_ptr        = &this->state_data_vec[this->num_y];

    // Q has shape of (n_stages + 1, num_y)
    // y = y_old + Q dot p.
    switch (this->solution_ptr->integrator_method)
    {
    case ODEMethod::RK23:
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t Q_stride = this->Q_order * y_i;
            // P=0
            // Initialize dot product
            double cumulative_prod = step_factor;
            double temp_double = Q_ptr[Q_stride] * cumulative_prod;
            // P=1
            cumulative_prod *= step_factor;
            temp_double += Q_ptr[Q_stride + 1] * cumulative_prod;
            // P=2
            cumulative_prod *= step_factor;
            temp_double += Q_ptr[Q_stride + 2] * cumulative_prod;

            // Finally multiply by step
            temp_double *= this->step;

            y_interp_ptr[y_i] = y_stored_ptr[y_i] + temp_double;
        }
        break;

    case ODEMethod::RK45:
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t Q_stride = this->Q_order * y_i;
            // P=0
            double cumulative_prod = step_factor;
            double temp_double = Q_ptr[Q_stride] * cumulative_prod;
            // P=1
            cumulative_prod *= step_factor;
            temp_double += Q_ptr[Q_stride + 1] * cumulative_prod;
            // P=2
            cumulative_prod *= step_factor;
            temp_double += Q_ptr[Q_stride + 2] * cumulative_prod;
            // P=3
            cumulative_prod *= step_factor;
            temp_double += Q_ptr[Q_stride + 3] * cumulative_prod;
            
            // Finally multiply by step
            temp_double *= this->step;

            y_interp_ptr[y_i] = y_stored_ptr[y_i] + temp_double;
        }
        break;

    case ODEMethod::DOP853:
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t Q_stride = this->Q_order * y_i;
            // This method is different from RK23 and RK45
            // Q is the reverse of SciPy's "F". The size of Q is (Interpolator power (Q_order), num_y)
            // DOP853 interp power is 7
            // This dense output does an alternating multiplier where even values of P_i are multiplied by the step factor.
            // Odd values are multiplied by 1 - step factor.

            // P=0
            double temp_double = Q_ptr[Q_stride];
            temp_double *= step_factor;
            // P=1
            temp_double += Q_ptr[Q_stride + 1];
            temp_double *= (1.0 - step_factor);
            // P=2
            temp_double += Q_ptr[Q_stride + 2];
            temp_double *= step_factor;
            // P=3
            temp_double += Q_ptr[Q_stride + 3];
            temp_double *= (1.0 - step_factor);
            // P=4
            temp_double += Q_ptr[Q_stride + 4];
            temp_double *= step_factor;
            // P=5
            temp_double += Q_ptr[Q_stride + 5];
            temp_double *= (1.0 - step_factor);
            // P=6
            temp_double += Q_ptr[Q_stride + 6];
            temp_double *= step_factor;

            y_interp_ptr[y_i] = y_stored_ptr[y_i] + temp_double;
        }
        break;

    case ODEMethod::RADAU:
        /* The collocation polynomial is evaluated directly, without the extra factor of the step
           size that the explicit Runge-Kutta interpolants carry. */
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t Q_stride = this->Q_order * y_i;
            // P=0
            double cumulative_prod = step_factor;
            double temp_double = Q_ptr[Q_stride] * cumulative_prod;
            // P=1
            cumulative_prod *= step_factor;
            temp_double += Q_ptr[Q_stride + 1] * cumulative_prod;
            // P=2
            cumulative_prod *= step_factor;
            temp_double += Q_ptr[Q_stride + 2] * cumulative_prod;

            y_interp_ptr[y_i] = y_stored_ptr[y_i] + temp_double;
        }
        break;

    case ODEMethod::BDF:
        /* SciPy's `BdfDenseOutput`: the backward differences are evaluated at
           x_j = (t_interp - (t_now - step * j)) / (step * (j + 1)) for j = 0 to order - 1, with
           the products accumulated so that p_j = x_0 * x_1 * ... * x_j.
           Q holds D[1] through D[order] and y_stored holds D[0]. */
        if (this->step == 0.0) [[unlikely]]
        {
            std::memcpy(y_interp_ptr, y_stored_ptr, sizeof(double) * this->num_y);
            break;
        }
        else
        {
            // The polynomial coefficients do not depend on y so build them once.
            double p_products[BDF_MAX_ORDER];
            double cumulative_prod = 1.0;
            for (size_t P_i = 0; P_i < this->Q_order; P_i++)
            {
                const double P_i_dbl = (double)P_i;
                cumulative_prod *= (t_interp - (this->t_now - this->step * P_i_dbl)) / (this->step * (P_i_dbl + 1.0));
                p_products[P_i] = cumulative_prod;
            }

            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                const size_t Q_stride = this->Q_order * y_i;
                double temp_double = 0.0;
                for (size_t P_i = 0; P_i < this->Q_order; P_i++)
                {
                    temp_double += Q_ptr[Q_stride + P_i] * p_products[P_i];
                }
                y_interp_ptr[y_i] = y_stored_ptr[y_i] + temp_double;
            }
        }
        break;

    case ODEMethod::LSODA:
        /* SciPy's `LsodaDenseOutput`: the Nordsieck history array is a Taylor series about the
           end of the step, y = sum_j yh[:, j] * ((t_interp - t_now) / step) ** j.
           Q holds columns 1 through `Q_order` and y_stored holds column 0. */
        if (this->step == 0.0) [[unlikely]]
        {
            std::memcpy(y_interp_ptr, y_stored_ptr, sizeof(double) * this->num_y);
            break;
        }
        else
        {
            const double step_factor = (t_interp - this->t_now) / this->step;

            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                const size_t Q_stride = this->Q_order * y_i;
                // Horner's method over the Taylor coefficients.
                double temp_double = Q_ptr[Q_stride + this->Q_order - 1];
                for (size_t P_i = this->Q_order - 1; P_i-- > 0; )
                {
                    temp_double = temp_double * step_factor + Q_ptr[Q_stride + P_i];
                }
                y_interp_ptr[y_i] = y_stored_ptr[y_i] + temp_double * step_factor;
            }
        }
        break;

    [[unlikely]] default:
        // Don't know the model. Just return the input.
        std::memcpy(y_interp_ptr, y_stored_ptr, sizeof(double) * this->num_y);
        break;
    }

    if (this->solution_ptr->capture_extra)
    {
        CySolverBase* solver_ptr = this->solution_ptr->solver_uptr.get();

        if (solver_ptr)
        {
            // We have interpolated the dependent y-values but have not handled any extra outputs
            // We can not use the RK (or any other integration method's) fancy interpolation because extra outputs are
            // not included in the, for example, Q matrix building process.
            // TODO: Perhaps we could include them in that? 
            // For now, we will make an additional call to the diffeq using the y0 we just found above and t_interp.
            
            size_t num_dy = solver_ptr->num_dy;

            // We will be overwriting the solver's now variables so tell it to store a copy that it can be restored back to.
            solver_ptr->offload_to_temp();

            // Load new values into t and y
            std::memcpy(solver_ptr->y_now_ptr, y_interp_ptr, sizeof(double) * this->num_y);
            solver_ptr->t_now = t_interp;
            
            // Call diffeq to update dy_now pointer
            solver_ptr->diffeq(solver_ptr);

            // Capture extra output and add to the y_interp_ptr array
            // We already have y interpolated from above so start at num_y
            for (size_t i = this->num_y; i < num_dy; i++)
            {
                y_interp_ptr[i] = solver_ptr->dy_now_ptr[i];
            }

            // Reset CySolver state to what it was before
            solver_ptr->load_back_from_temp();
        }
    }
}