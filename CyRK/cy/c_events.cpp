#include "c_events.hpp"


/* ========================================================================= */
/* ========================  Event Wrapper  ================================ */
/* ========================================================================= */
// Constructors and Descructors don't do much.
Event::Event()
{

}

Event::Event(
        EventFunc event_func,
        size_t max_allowed_,
        int direction_):
            check(event_func),
            max_allowed(max_allowed_),
            status(CyrkErrorCodes::NO_ERROR),
            direction(direction_),
            initialized(true)
{
    if (!this->check)
    {
        this->initialized = false;
        this->status = CyrkErrorCodes::EVENT_SETUP_FAILED;
    }
}


Event::~Event()
{
    
}

// C++ / Cython Setup
CyrkErrorCodes Event::setup(
        EventFunc event_func,
        size_t max_allowed_,
        int direction_)
{
    // Reset initialized flag.
    this->initialized = false;

    if (!event_func)
    {
        this->status = CyrkErrorCodes::EVENT_SETUP_FAILED;
        return CyrkErrorCodes::EVENT_SETUP_FAILED;
    }
    this->check = event_func;
    this->direction = direction_;
    this->max_allowed = max_allowed_;

    // Everything seems good.
    this->current_count = 0;
    this->last_root   = 0.0;
    this->is_active   = false;
    this->initialized = true;
    this->status = CyrkErrorCodes::NO_ERROR;
    return CyrkErrorCodes::NO_ERROR;
}
