#pragma once

#include <cmath>
#include <limits>

// Integration Constants
// Multiply steps computed from asymptotic behaviour of errors by this.
const double SAFETY = 0.9;  // Error coefficient factor (1 == use calculated error; < 1 means be conservative).
const double MIN_FACTOR = 0.2;  // Minimum allowed decrease in a step size.
const double MAX_FACTOR = 10.;  // Maximum allowed increase in a step size.
const double INF = std::numeric_limits<double>::infinity();
const double MAX_STEP = INF;
const double EPS = std::numeric_limits<double>::epsilon();
const double EPS_10 = EPS * 10.0;
const double EPS_100 = EPS * 100.0;
const size_t MAX_SIZET_SIZE = std::numeric_limits<size_t>::max();
const size_t MAX_INT_SIZE = std::numeric_limits<int>::max();


// Memory management constants
// Assume that a cpu has a L1 of 300KB.Say that this progam will have access to 75 % of that total.
const double CPU_CACHE_SIZE = 0.75 * 300000.0;
// Number of entities we can fit into that size is based on the size of double(or double complex)
const double MAX_ARRAY_PREALLOCATE_SIZE_DBL = 600000.0;
const double MIN_ARRAY_PREALLOCATE_SIZE = 10.0;
const double ARRAY_PREALLOC_TABS_SCALE = 1000.0;  // A delta_t_abs higher than this value will start to grow array size.
const double ARRAY_PREALLOC_RTOL_SCALE = 1.0e-5;  // A rtol lower than this value will start to grow array size.

// Solution parameters
const double DYNAMIC_GROWTH_RATE = 1.618;
// To learn why the golden ratio is used, read this:
// https://stackoverflow.com/questions/1100311/what-is-the-ideal-growth-rate-for-a-dynamically-allocated-array
const double SIZE_MAX_DBL = 0.99 * SIZE_MAX;


typedef void (*DiffeqFuncType)(double*, double, double*, double*);

void find_max_num_steps(
    size_t num_y,
    size_t num_extra,
    size_t max_num_steps,
    size_t max_ram_MB,
    bool capture_extra,
    bool* user_provided_max_num_steps,
    size_t* max_num_steps_touse);

size_t find_expected_size(
    size_t num_y,
    size_t num_extra,
    double t_delta_abs,
    double rtol_min,
    bool capture_extra);
