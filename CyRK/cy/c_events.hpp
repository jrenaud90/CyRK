#pragma once

#include <limits>

#include "common.hpp"

typedef double (*EventFunc)(double, double*, char*);

constexpr size_t MAX_ALLOWED_SYS = std::numeric_limits<size_t>::max();


class Event
{
public:    
    EventFunc check     = nullptr;
    // By default events are always recorded with no limit and no termination.
    size_t max_allowed  = MAX_ALLOWED_SYS;
    int termination_int = 0;
    CyrkErrorCodes status = CyrkErrorCodes::UNINITIALIZED_CLASS;
    bool initialized    = false;

    Event();
    Event(
        EventFunc event_func,
        size_t max_allowed_ = MAX_ALLOWED_SYS,
        int termination_int_ = 0);
    virtual ~Event();

    CyrkErrorCodes setup(
        EventFunc event_func,
        size_t max_allowed_ = MAX_ALLOWED_SYS,
        int termination_int_ = 0);

protected:
};