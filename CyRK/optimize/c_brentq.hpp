#pragma once

#include <vector>
#include "c_common.hpp"
#include "c_events.hpp"
#include "dense.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

double c_brentq(
        EventFuncWithInst func,
        double xa,
        double xb,
        double xtol,
        double rtol,
        size_t iter,
        std::vector<char>& func_data_vec,
        OptimizeInfo* solver_stats,
        Event* event_ptr,
        CySolverDense* dense_func = nullptr);
