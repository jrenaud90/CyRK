#include <stddef.h>   // `size_t`
#include <stdbool.h>  // `bool`
#include <math.h>     // `fmin`, `fmax`, `fabs`
#include <stdio.h>
// Create a fake struct to trick C into accepting the CySolver class (which contains the diffeq method)
struct CySolverStruct {
    char empty;
};


int rk_step_cf(
        // Pointer to differential equation
        void (*diffeq_ptr)(struct CySolverStruct*),
        // Pointer to the CySolver instance
        struct CySolverStruct* cysolver_inst,

        // t-related variables
        double t_end,
        bool direction_flag,
        double direction_inf,

        // y-related variables
        size_t y_size,
        double y_size_dbl,
        double y_size_sqrt,

        // Pointers to class attributes that can change during rk_step call.
        double* restrict t_now_ptr,
        double* restrict y_ptr,
        double* restrict dy_ptr,
        double* restrict t_old_ptr,
        double* restrict y_old_ptr,
        double* restrict dy_old_ptr,
        double* restrict step_size_ptr,
        char* restrict status_ptr,

        // Integration tolerance variables and pointers
        double* restrict atols_ptr,
        double* restrict rtols_ptr,
        double max_step,

        // RK specific variables and pointers
        unsigned char rk_method,
        size_t rk_n_stages,
        size_t rk_n_stages_plus1,
        size_t len_Acols,
        size_t len_C,
        double* restrict A_ptr,
        double* restrict B_ptr,
        double* restrict C_ptr,
        double* restrict K_ptr,
        double* restrict E_ptr,
        double* restrict E3_ptr,
        double* restrict E5_ptr,
        double error_expo,
        double min_step_factor,
        double max_step_factor,
        double error_safety
        ){
    /**
     * Performs a Runge-Kutta step calculation including local error determination. 
    */

    // Initialize step variables
    double min_step, step, step_factor, time_tmp, t_delta_check;
    double scale, temp_double;
    double error_norm, error_dot_1, error_pow;
    bool step_accepted, step_rejected, step_error;

    // Store values from pointers so that they do not have to be dereferenced multiple times
    double t_now = *t_now_ptr;
    double t_old = *t_old_ptr;
    double step_size = *step_size_ptr;

    // Run RK integration step
    // Determine step size based on previous loop
    // Find minimum step size based on the value of t (less floating point numbers between numbers when t is large)
    min_step = 10. * fabs(nextafter(t_old, direction_inf) - t_old);
    // Look for over/undershoots in previous step size
    if (step_size > max_step) {
        step_size = max_step;
    } else if (step_size < min_step) {
        step_size = min_step;
    }

    // Determine new step size
    step_accepted = false;
    step_rejected = false;
    step_error    = false;

    // Optimization variables
    // Define a very specific A (Row 1; Col 0) now since it is called consistently and does not change.
    double A_at_10 = A_ptr[1 * len_Acols + 0];

    // !! Step Loop
    while (!step_accepted) {

        // Check if step size is too small
        // This will cause integration to fail: step size smaller than spacing between numbers
        if (step_size < min_step) {
            step_error  = true;
            *status_ptr = -1;
            break;
        }

        // Move time forward for this particular step size
        if (direction_flag) {
            step          = step_size;
            t_now         = t_old + step;
            t_delta_check = t_now - t_end;
        } else {
            step          = -step_size;
            t_now         = t_old + step;
            t_delta_check = t_end - t_now;
        }

        // Check that we are not at the end of integration with that move
        if (t_delta_check > 0.0) {
            t_now = t_end;

            // If we are, correct the step so that it just hits the end of integration.
            step = t_now - t_old;
            if (direction_flag){
                step_size = step;
            } else {
                step_size = -step;
            }
        }

        // !! Calculate derivative using RK method

        // t_now must be updated for each loop of s in order to make the diffeq method calls.
        // But we need to return to its original value later on. Store in temp variable.
        time_tmp = t_now;

        for (size_t s = 1; s < len_C; s++) {
            // Find the current time based on the old time and the step size.
            t_now = t_old + C_ptr[s] * step;
            // Update the value stored at the t_now pointer so it can be used in the diffeq method.
            *t_now_ptr = t_now;

            // Dot Product (K, a) * step
            if (s == 1) {
                for (size_t i = 0; i < y_size; i++) {
                    // Set the first column of K
                    temp_double = dy_old_ptr[i];
                    // K[0, :] == first part of the array
                    K_ptr[i] = temp_double;

                    // Calculate y_new for s==1
                    y_ptr[i] = y_old_ptr[i] + (temp_double * A_at_10 * step);
                }
            } else {
                for (size_t j = 0; j < s; j++) {
                    temp_double = A_ptr[s * len_Acols + j] * step;
                    for (size_t i = 0; i < y_size; i++) {
                        if (j == 0){
                            // Initialize
                            y_ptr[i] = y_old_ptr[i];
                        }
                        y_ptr[i] += K_ptr[j * y_size + i] * temp_double;
                    }
                }
            }
            // Call diffeq method to update K with the new dydt
            // This will use the now updated values at y_ptr and t_now_ptr. It will update values at dy_ptr.
            diffeq_ptr(cysolver_inst);

            // Update K based on the new dy values.
            for (size_t i = 0; i < y_size; i++) {
                K_ptr[s * y_size + i] = dy_ptr[i];
            }
        }

        // Restore t_now to its previous value.
        t_now = time_tmp;
        // Update the pointer.
        *t_now_ptr = t_now;
        
        // Dot Product (K, B) * step
        for (size_t j = 0; j < rk_n_stages; j++) {
            temp_double = B_ptr[j] * step;
            // We do not use rk_n_stages_plus1 here because we are chopping off the last row of K to match
            //  the shape of B.
            for (size_t i = 0; i < y_size; i++) {
                if (j == 0) {
                    // Initialize
                    y_ptr[i] = y_old_ptr[i];
                }
                y_ptr[i] += K_ptr[j * y_size + i] * temp_double;
            }
        }
        
        // Find final dydt for this timestep
        // This will use the now final values at y_ptr and t_now_ptr. It will update values at dy_ptr.
        diffeq_ptr(cysolver_inst);

        // Check how well this step performed by calculating its error.
        if (rk_method == 2) {
             double error_norm3, error_norm5, error_dot_2, error_denom; 
            // Calculate Error for DOP853
            // Dot Product (K, E5) / scale and Dot Product (K, E3) * step / scale
            error_norm3 = 0.0;
            error_norm5 = 0.0;
            for (size_t i = 0; i < y_size; i++) {
                // Find scale of y for error calculations
                scale = (atols_ptr[i] + fmax(fabs(y_old_ptr[i]), fabs(y_ptr[i])) * rtols_ptr[i]);

                // Set last array of K equal to dydt
                K_ptr[rk_n_stages * y_size + i] = dy_ptr[i];

                // Initialize
                error_dot_1 = 0.0;
                error_dot_2 = 0.0;
                
                for (size_t j = 0; j < rk_n_stages_plus1; j++) {
                    temp_double = K_ptr[j * y_size + i];
                    error_dot_1 += temp_double * E3_ptr[j];
                    error_dot_2 += temp_double * E5_ptr[j];
                }
                // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
                // TODO: This will need to change if CySolver ever accepts complex numbers
                // error_norm3_abs = fabs(error_dot_1)
                // error_norm5_abs = fabs(error_dot_2)
                error_dot_1 /= scale;
                error_dot_2 /= scale;

                error_norm3 += (error_dot_1 * error_dot_1);
                error_norm5 += (error_dot_2 * error_dot_2);
            }
            // Check if errors are zero
            if ((error_norm5 == 0.0) && (error_norm3) == 0.0) {
                error_norm = 0.0;
            } else {
                error_denom = error_norm5 + 0.01 * error_norm3;
                error_norm = step_size * error_norm5 / sqrt(error_denom * y_size_dbl);
            }
        } else {
            // Calculate Error for RK23 and RK45
            // Dot Product (K, E) * step / scale
            error_norm = 0.0;
            for (size_t i = 0; i < y_size; i++) {
                // Find scale of y for error calculations
                scale = (atols_ptr[i] + fmax(fabs(y_old_ptr[i]), fabs(y_ptr[i])) * rtols_ptr[i]);

                // Set last array of K equal to dydt
                K_ptr[rk_n_stages * y_size + i] = dy_ptr[i];
                
                // Initialize
                error_dot_1 = 0.0;

                for (size_t j = 0; j < rk_n_stages_plus1; j++) {
                    error_dot_1 += K_ptr[j * y_size + i] * E_ptr[j];
                }

                // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
                // TODO: This will need to change if CySolver ever accepts complex numbers
                // error_norm_abs = fabs(error_dot_1)
                error_dot_1 *= (step / scale);

                error_norm += (error_dot_1 * error_dot_1);
            }
            error_norm = sqrt(error_norm) / y_size_sqrt;
        }

        // Check the size of the error
        if (error_norm < 1.0) {
            // We found our step size because the error is low!
            // Update this step for the next time loop
            if (error_norm == 0.0) {
                step_factor = max_step_factor;
            } else {
                error_pow = pow(error_norm, -error_expo);
                step_factor = fmin(max_step_factor, error_safety * error_pow);
            }

            if (step_rejected) {
                // There were problems with this step size on the previous step loop. Make sure factor does
                //   not exasperate them.
                step_factor = fmin(step_factor, 1.);
            }

            // Update step size
            step_size *= step_factor;
            step_accepted = true;
        } else {
            // Error is still large. Keep searching for a better step size.
            error_pow = pow(error_norm, -error_expo);
            step_size *= fmax(min_step_factor, error_safety * error_pow);
            step_rejected = true;
        }
    }

    // Update status depending if there were any errors.
    if (step_error) {
        // Issue with step convergence
        *status_ptr = -1;
    } else if (!step_accepted) {
        // Issue with step convergence
        *status_ptr = -7;
    }

    // End of RK step. 
    // Update "old" pointers
    *t_old_ptr = t_now;
    for (size_t i = 0; i < y_size; i++) {
        y_old_ptr[i]  = y_ptr[i];
        dy_old_ptr[i] = dy_ptr[i];
    }

    // Update any other pointers
    *step_size_ptr = step_size;

    return 0;
}
