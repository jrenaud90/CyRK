/* Helper functions and global constants */

#include "common.hpp"


void round_to_2(size_t &initial_value)
{
    /* Rounds the initial value to the nearest power of 2 */
    // Method is the fastest for 64-bit numbers
    initial_value--;
    initial_value |= initial_value >> 1;
    initial_value |= initial_value >> 2;
    initial_value |= initial_value >> 4;
    initial_value |= initial_value >> 8;
    initial_value |= initial_value >> 16;
    initial_value |= initial_value >> 32;
    initial_value++;
}

MaxNumStepsOutput find_max_num_steps(
    const size_t num_y,
    const size_t num_extra,
    const size_t max_num_steps,
    const size_t max_ram_MB)
{
    /* Determines the maximum number of steps (max size of time domain) that the integrator is allowed to take. */

    // Determine the maximum number of steps permitted during integration run.
    // Then divide by number of dependnet and extra variables that will be stored. The extra "1" is for the time domain.
    double max_num_steps_ram_dbl = max_ram_MB * (1000.0 * 1000.0) / (sizeof(double) * (1.0 + num_y + num_extra));
    size_t max_num_steps_ram = (size_t)std::floor(max_num_steps_ram_dbl);

    MaxNumStepsOutput output(false, max_num_steps_ram);

    if (max_num_steps > 0)
    {
        // User provided a maximum number of steps; parse their input.
        if (max_num_steps > max_num_steps_ram)
        {
            output.max_num_steps = max_num_steps_ram;
        }
        else
        {
            output.user_provided_max_num_steps = true;
            output.max_num_steps = max_num_steps;
        }
    }

    // Make sure that max number of steps does not exceed size_t limit
    if (output.max_num_steps > (MAX_SIZET_SIZE / 10))
    {
        output.max_num_steps = (MAX_SIZET_SIZE / 10);
    }

    return output;
}


size_t find_expected_size(
    const size_t num_y,
    const size_t num_extra,
    const double t_delta_abs,
    const double rtol_min)
    /* Finds an expected size for storage arrays (length of time domain) that is suitable to the provided problem */
{
    // Pick starting value that works with a lot of problems.
    double temp_expected_size = 256;
    // If t_delta_abs is very large or rtol is very small, then we may need more.
    temp_expected_size = std::fmax(temp_expected_size, std::fmax(t_delta_abs / ARRAY_PREALLOC_TABS_SCALE, ARRAY_PREALLOC_RTOL_SCALE / rtol_min));
    // Fix values that are very small / large
    temp_expected_size = std::fmax(temp_expected_size, MIN_ARRAY_PREALLOCATE_SIZE);
    double max_expected = num_extra ? MAX_ARRAY_PREALLOCATE_SIZE_DBL / (num_y + num_extra) : MAX_ARRAY_PREALLOCATE_SIZE_DBL / num_y;

    temp_expected_size = fmin(temp_expected_size, max_expected);
    size_t expected_size_to_use = (size_t)std::floor(temp_expected_size);
    
    return expected_size_to_use;
}
