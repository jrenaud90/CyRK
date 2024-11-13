
#include "dense.hpp"

#include <cstdio>

CySolverDense::CySolverDense(
        int integrator_int,
        double t_old,
        double t_now,
        double* y_in_ptr,
        unsigned int num_y,
        unsigned int num_extra,
        unsigned int Q_order,
        CySolverBase* cysolver_instance_ptr,
        std::function<void (CySolverBase *)> cysolver_diffeq_ptr,
        double* cysolver_t_now_ptr,
        double* cysolver_y_now_ptr,
        double* cysolver_dy_now_ptr
        ) :
            integrator_int(integrator_int),
            num_y(num_y),
            num_extra(num_extra),
            cysolver_instance_ptr(cysolver_instance_ptr),
            cysolver_diffeq_ptr(cysolver_diffeq_ptr),
            cysolver_t_now_ptr(cysolver_t_now_ptr),
            cysolver_y_now_ptr(cysolver_y_now_ptr),
            cysolver_dy_now_ptr(cysolver_dy_now_ptr),
            t_old(t_old),
            t_now(t_now),
            Q_order(Q_order)
{
    // Make a copy of the y_in pointer in this Dense interpolator's storage
    std::memcpy(this->y_stored_ptr, y_in_ptr, sizeof(double) * this->num_y);
    // Calculate step
    this->step = this->t_now - this->t_old;
}

void CySolverDense::call(double t_interp, double* y_intepret)
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
            double temp_double = this->Q_ptr[Q_stride] * cumulative_prod;
            // P=1
            cumulative_prod *= step_factor;
            temp_double += this->Q_ptr[Q_stride + 1] * cumulative_prod;
            // P=2
            cumulative_prod *= step_factor;
            temp_double += this->Q_ptr[Q_stride + 2] * cumulative_prod;

            // Finally multiply by step
            temp_double *= this->step;

            y_intepret[y_i] = this->y_stored_ptr[y_i] + temp_double;
        }
        break;

    case 1:
        // RK45
        for (unsigned int y_i = 0; y_i < this->num_y; y_i++)
        {
            const unsigned int Q_stride = this->Q_order * y_i;
            // P=0
            double cumulative_prod = step_factor;
            double temp_double = this->Q_ptr[Q_stride] * cumulative_prod;
            // P=1
            cumulative_prod *= step_factor;
            temp_double += this->Q_ptr[Q_stride + 1] * cumulative_prod;
            // P=2
            cumulative_prod *= step_factor;
            temp_double += this->Q_ptr[Q_stride + 2] * cumulative_prod;
            // P=3
            cumulative_prod *= step_factor;
            temp_double += this->Q_ptr[Q_stride + 3] * cumulative_prod;
            
            // Finally multiply by step
            temp_double *= this->step;

            y_intepret[y_i] = this->y_stored_ptr[y_i] + temp_double;
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
            double temp_double = this->Q_ptr[Q_stride];
            temp_double *= step_factor;
            // P=1
            temp_double += this->Q_ptr[Q_stride + 1];
            temp_double *= (1.0 - step_factor);
            // P=2
            temp_double += this->Q_ptr[Q_stride + 2];
            temp_double *= step_factor;
            // P=3
            temp_double += this->Q_ptr[Q_stride + 3];
            temp_double *= (1.0 - step_factor);
            // P=4
            temp_double += this->Q_ptr[Q_stride + 4];
            temp_double *= step_factor;
            // P=5
            temp_double += this->Q_ptr[Q_stride + 5];
            temp_double *= (1.0 - step_factor);
            // P=6
            temp_double += this->Q_ptr[Q_stride + 6];
            temp_double *= step_factor;

            y_intepret[y_i] = this->y_stored_ptr[y_i] + temp_double;
        }
        break;

    [[unlikely]] default:
        // Don't know the model. Just return the input.
        std::memcpy(y_intepret, this->y_stored_ptr, sizeof(double) * this->num_y);
        break;
    }

    if (this->num_extra > 0)
    {
        // We have interpolated the dependent y-values but have not handled any extra outputs
        // We can not use the RK (or any other integration method's) fancy interpolation because extra outputs are
        // not included in the, for example, Q matrix building process.
        // TODO: Perhaps we could include them in that? 
        // For now, we will make an additional call to the diffeq using the y0 we just found above and t_interp.
        
        size_t num_dy = this->num_y + this->num_extra;

        // Store a copy of dy_now, t_now, and y_now into old vectors so we can make the call non destructively.
        // y array
        double y_tmp[Y_LIMIT];
        double* y_tmp_ptr = &y_tmp[0];
        memcpy(y_tmp_ptr, this->cysolver_y_now_ptr, this->num_y);
        // dy array
        double dy_tmp[DY_LIMIT];
        double* dy_tmp_ptr = &dy_tmp[0];
        memcpy(dy_tmp_ptr, this->cysolver_dy_now_ptr, num_dy);
        // t
        double t_tmp = cysolver_t_now_ptr[0];

        // Load new values into t and y
        memcpy(this->cysolver_y_now_ptr, y_intepret, this->num_y);
        cysolver_t_now_ptr[0] = t_interp;
        
        // Call diffeq to update dy_now pointer
        this->cysolver_diffeq_ptr(this->cysolver_instance_ptr);

        // Capture extra output and add to the y_intepret array
        // We already have y interpolated from above so start at num_y
        for (size_t i = this->num_y; i < num_dy; i++)
        {
            y_intepret[i] = this->cysolver_dy_now_ptr[i];
        }

        // Reset CySolver state to what it was before
        cysolver_t_now_ptr[0] = t_tmp;
        memcpy(this->cysolver_y_now_ptr, y_tmp_ptr, num_dy);
        memcpy(this->cysolver_dy_now_ptr, dy_tmp_ptr, num_dy);
    }
}