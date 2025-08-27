#pragma once
#include <vector>
#include "c_common.hpp"
#include "c_events.hpp"
#include "dense.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct OptimizeInfo {
    size_t funcalls = 0;
    size_t iterations = 0;
    CyrkErrorCodes error_num = CyrkErrorCodes::UNSET_ERROR_CODE;
};

double c_brentq(
        EventFunc func,
        double xa,
        double xb,
        double xtol,
        double rtol,
        size_t iter,
        std::vector<char>& func_data_vec,
        OptimizeInfo* solver_stats,
        CySolverDense* dense_func = nullptr);
