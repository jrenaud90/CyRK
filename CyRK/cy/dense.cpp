
#include "dense.hpp"
#include "cysolver.hpp"

// Constructors
CySolverDense::CySolverDense(
        int integrator_int,
        std::shared_ptr<CySolverBase> solver_sptr,
        bool set_state
        ) :
            integrator_int(integrator_int),
            num_y(solver_sptr->num_y),
            num_extra(solver_sptr->num_extra),
            solver_sptr(solver_sptr)
{
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
    this->t_old = solver_sptr->t_old;
    this->t_now = solver_sptr->t_now;
    // Make a copy of the y_in pointer in this Dense interpolator's storage
    std::memcpy(&this->y_stored[0], &this->solver_sptr->y_old[0], sizeof(double) * this->num_y);
    // Calculate step
    this->step = this->t_now - this->t_old;

    // Perform setup for specific integrator types
    switch (this->integrator_int)
    {
        // RK Methods
        case 0:
        case 1:
        case 2:
            // Tell the RK solver to set the values of the Q array.
            this->solver_sptr->set_Q_array(&this->Q[0], &this->Q_order);
            break;
        
        default:
            break;
    }
}

void CySolverDense::call(double t_interp, double* y_interp_ptr)
{
    double step_factor = (t_interp - this->t_old) / this->step;

    // SciPy Step:: p = np.tile(x, self.order + 1) (scipy order is Q_order - 1)

    // Q has shape of (n_stages + 1, num_y)
    // y = y_old + Q dot p.
    switch (this->integrator_int)
    {
    case 0:
        // RK23
        for (unsigned int y_i = 0; y_i < this->num_y; y_i++)
        {
            unsigned int Q_stride = this->Q_order * y_i;
            // P=0
            // Initialize dot product
            double cumulative_prod = step_factor;
            double temp_double = this->Q[Q_stride] * cumulative_prod;
            // P=1
            cumulative_prod *= step_factor;
            temp_double += this->Q[Q_stride + 1] * cumulative_prod;
            // P=2
            cumulative_prod *= step_factor;
            temp_double += this->Q[Q_stride + 2] * cumulative_prod;

            // Finally multiply by step
            temp_double *= this->step;

            y_interp_ptr[y_i] = this->y_stored[y_i] + temp_double;
        }
        break;

    case 1:
        // RK45
        for (unsigned int y_i = 0; y_i < this->num_y; y_i++)
        {
            const unsigned int Q_stride = this->Q_order * y_i;
            // P=0
            double cumulative_prod = step_factor;
            double temp_double = this->Q[Q_stride] * cumulative_prod;
            // P=1
            cumulative_prod *= step_factor;
            temp_double += this->Q[Q_stride + 1] * cumulative_prod;
            // P=2
            cumulative_prod *= step_factor;
            temp_double += this->Q[Q_stride + 2] * cumulative_prod;
            // P=3
            cumulative_prod *= step_factor;
            temp_double += this->Q[Q_stride + 3] * cumulative_prod;
            
            // Finally multiply by step
            temp_double *= this->step;

            y_interp_ptr[y_i] = this->y_stored[y_i] + temp_double;
        }
        break;

    case 2:
        // DOP853
        for (unsigned int y_i = 0; y_i < this->num_y; y_i++)
        {
            const unsigned int Q_stride = this->Q_order * y_i;
            // This method is different from RK23 and RK45
            // Q is the reverse of SciPy's "F". The size of Q is (Interpolator power (Q_order), num_y)
            // DOP853 interp power is 7
            // This dense output does an alternating multiplier where even values of P_i are multiplied by the step factor.
            // Odd values are multiplied by 1 - step factor.

            // P=0
            double temp_double = this->Q[Q_stride];
            temp_double *= step_factor;
            // P=1
            temp_double += this->Q[Q_stride + 1];
            temp_double *= (1.0 - step_factor);
            // P=2
            temp_double += this->Q[Q_stride + 2];
            temp_double *= step_factor;
            // P=3
            temp_double += this->Q[Q_stride + 3];
            temp_double *= (1.0 - step_factor);
            // P=4
            temp_double += this->Q[Q_stride + 4];
            temp_double *= step_factor;
            // P=5
            temp_double += this->Q[Q_stride + 5];
            temp_double *= (1.0 - step_factor);
            // P=6
            temp_double += this->Q[Q_stride + 6];
            temp_double *= step_factor;

            y_interp_ptr[y_i] = this->y_stored[y_i] + temp_double;
        }
        break;

    [[unlikely]] default:
        // Don't know the model. Just return the input.
        std::memcpy(y_interp_ptr, &this->y_stored[0], sizeof(double) * this->num_y);
        break;
    }

    if (this->num_extra > 0)
    {
        if (this->solver_sptr.get())
        {
            // We have interpolated the dependent y-values but have not handled any extra outputs
            // We can not use the RK (or any other integration method's) fancy interpolation because extra outputs are
            // not included in the, for example, Q matrix building process.
            // TODO: Perhaps we could include them in that? 
            // For now, we will make an additional call to the diffeq using the y0 we just found above and t_interp.
            
            size_t num_dy = this->num_y + this->num_extra;

            // We will be overwriting the solver's now variables so tell it to store a copy that it can be restored back to.
            this->solver_sptr->offload_to_temp();

            // Load new values into t and y
            std::memcpy(&this->solver_sptr->y_now[0], y_interp_ptr, sizeof(double) * this->num_y);
            this->solver_sptr->t_now = t_interp;
            
            // Call diffeq to update dy_now pointer
            this->solver_sptr->diffeq(this->solver_sptr.get());

            // Capture extra output and add to the y_interp_ptr array
            // We already have y interpolated from above so start at num_y
            for (size_t i = this->num_y; i < num_dy; i++)
            {
                y_interp_ptr[i] = this->solver_sptr->dy_now[i];
            }

            // Reset CySolver state to what it was before
            this->solver_sptr->load_back_from_temp();
        }
        else
        {
            throw std::exception("Can not complete interpolation for extra outputs because solver has been deconstructed.");
        }
    }
}