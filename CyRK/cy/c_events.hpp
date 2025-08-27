#pragma once

#include <vector>
#include <limits>

#include "c_common.hpp"

typedef double (*EventFunc)(double, double*, char*);

constexpr size_t MAX_ALLOWED_SYS = std::numeric_limits<size_t>::max();


class Event
{
public:    
    EventFunc check       = nullptr;
    // By default events are always recorded with no limit and no termination.
    size_t max_allowed    = MAX_ALLOWED_SYS;
    size_t current_count  = 0;
    CyrkErrorCodes status = CyrkErrorCodes::UNINITIALIZED_CLASS;
    double last_root      = 0.0;
    int direction         = 0;
    bool is_active        = false;
    bool initialized      = false;

    std::vector<double> y_at_root_vec = std::vector<double>();

    Event();
    Event(
        EventFunc event_func,
        size_t max_allowed_ = MAX_ALLOWED_SYS,
        int direction_ = 0);
    virtual ~Event();

    CyrkErrorCodes setup(
        EventFunc event_func,
        size_t max_allowed_ = MAX_ALLOWED_SYS,
        int direction_ = 0);

protected:
};