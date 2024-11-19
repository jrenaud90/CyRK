#include "dense.hpp"
#include "cysolver.hpp"

// Constructors
CySolverDense::CySolverDense(
        int integrator_int,
        CySolverBase* solver_ptr,
        bool set_state
        ) :
            integrator_int(integrator_int),
            num_y(solver_ptr->num_y),
            solver_ptr(solver_ptr)
{
    // Allocate memory for state vectors (memory allocation should not change since num_y does not change)
    this->num_y = this->solver_ptr->num_y;
    this->solver_ptr->set_Q_order(&this->Q_order);

    // Resize state vector based on dimensions. The state vector is a combination of the current y-values and Q
    // Q is a matrix of solver-specific parameters at the current time.
    // Q is defined by Q = K.T.dot(self.P)  K has shape of (n_stages + 1, num_y) so K.T has shape of (num_y, n_stages + 1)
    // P has shape of (4, 3) for RK23; (7, 4) for RK45.. So (n_stages + 1, Q_order)
    // So Q has shape of (num_y, q_order)
    // The max size of Q is (7) * num_y for DOP853
    // state vector is laid out as [y_vector, Q_matrix]
    this->state_data_vec.resize(this->num_y * (this->Q_order + 1));  // +1 is so we can store y_values in the first spot.

    // Populate values with current state
    if (set_state)
    {
        this->set_state();
    }
}

// Destructors
CySolverDense::~CySolverDense()
{

}

void CySolverDense::set_state()
{
    // Store time information
    this->t_old = solver_ptr->t_old;
    this->t_now = solver_ptr->t_now;
    
    // Calculate step
    this->step = this->t_now - this->t_old;

    // Make a copy of the y_in pointer in the state vector storage
    std::memcpy(&this->state_data_vec[0], &this->solver_ptr->y_old[0], sizeof(double) * this->num_y);

    // Tell the solver to populate the values of the Q matrix. 
    // Q starts at the num_y location of the state vector
    this->solver_ptr->set_Q_array(&this->state_data_vec[this->num_y]);
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
    switch (this->integrator_int)
    {
    case 0:
        // RK23
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

    case 1:
        // RK45
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

    case 2:
        // DOP853
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

    [[unlikely]] default:
        // Don't know the model. Just return the input.
        std::memcpy(y_interp_ptr, y_stored_ptr, sizeof(double) * this->num_y);
        break;
    }

    if (this->solver_ptr)
    {
        if (this->solver_ptr->num_extra > 0)
        {
            // We have interpolated the dependent y-values but have not handled any extra outputs
            // We can not use the RK (or any other integration method's) fancy interpolation because extra outputs are
            // not included in the, for example, Q matrix building process.
            // TODO: Perhaps we could include them in that? 
            // For now, we will make an additional call to the diffeq using the y0 we just found above and t_interp.
            
            size_t num_dy = this->solver_ptr->num_dy;

            // We will be overwriting the solver's now variables so tell it to store a copy that it can be restored back to.
            this->solver_ptr->offload_to_temp();

            // Load new values into t and y
            std::memcpy(&this->solver_ptr->y_now[0], y_interp_ptr, sizeof(double) * this->num_y);
            this->solver_ptr->t_now = t_interp;
            
            // Call diffeq to update dy_now pointer
            this->solver_ptr->diffeq(this->solver_ptr);

            // Capture extra output and add to the y_interp_ptr array
            // We already have y interpolated from above so start at num_y
            for (size_t i = this->num_y; i < num_dy; i++)
            {
                y_interp_ptr[i] = this->solver_ptr->dy_now[i];
            }

            // Reset CySolver state to what it was before
            this->solver_ptr->load_back_from_temp();
        }
    }
}