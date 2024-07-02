#include "common.hpp"


void find_max_num_steps(
    size_t num_y,
    size_t num_extra,
    size_t max_num_steps,
    size_t max_ram_MB,
    bool capture_extra,
    bool* user_provided_max_num_steps,
    size_t* max_num_steps_touse) {

    // Determine the maximum number of steps permitted during integration run.
    double max_num_steps_ram_dbl = max_ram_MB * (1000.0 * 1000.0);

    // Divide by number of dependnet and extra variables that will be stored. The extra "1" is for the time domain.
    if (capture_extra)
    {
        max_num_steps_ram_dbl /= (sizeof(double) * (1.0 + num_y + num_extra));
    }
    else {
        max_num_steps_ram_dbl /= (sizeof(double) * (1.0 + num_y));
    }
    size_t max_num_steps_ram = (size_t)std::floor(max_num_steps_ram_dbl);

    // Parse user-provided max number of steps
    user_provided_max_num_steps[0] = false;
    if (max_num_steps == 0)
    {
        // No user input; use ram-based value
        max_num_steps_touse[0] = max_num_steps_ram;
    }
    else {
        if (max_num_steps > max_num_steps_ram)
        {
            max_num_steps_touse[0] = max_num_steps_ram;
        }
        else {
            user_provided_max_num_steps[0] = true;
            max_num_steps_touse[0] = max_num_steps;
        }
    }

    // Make sure that max number of steps does not exceed size_t limit
    if (max_num_steps_touse[0] > (MAX_SIZET_SIZE / 10))
    {
        max_num_steps_touse[0] = (MAX_SIZET_SIZE / 10);
    }
}

size_t find_expected_size(
    size_t num_y,
    size_t num_extra,
    double t_delta_abs,
    double rtol_min,
    bool capture_extra)
{
    // Pick starting value that works with most problems
    double temp_expected_size = 500.0;
    // If t_delta_abs is very large or rtol is very small, then we may need more.
    temp_expected_size = std::fmax(temp_expected_size, std::fmax(t_delta_abs / ARRAY_PREALLOC_TABS_SCALE, ARRAY_PREALLOC_RTOL_SCALE / rtol_min));
    // Fix values that are very small / large
    temp_expected_size = std::fmax(temp_expected_size, MIN_ARRAY_PREALLOCATE_SIZE);
    double max_expected = MAX_ARRAY_PREALLOCATE_SIZE_DBL;
    if (capture_extra)
    {
        max_expected /= (num_y + num_extra);
    }
    else
    {
        max_expected /= num_y;

    }
    temp_expected_size = fmin(temp_expected_size, max_expected);
    size_t expected_size_to_use = (size_t)std::floor(temp_expected_size);
    
    return expected_size_to_use;
}