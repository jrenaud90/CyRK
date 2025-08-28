#include "c_events.hpp"


/* ========================================================================= */
/* ========================  Event Wrapper  ================================ */
/* ========================================================================= */
// Constructors and Deconstructors don't do much.
Event::Event()
{

}

Event::Event(
        EventFunc event_func,
        size_t max_allowed_,
        int direction_):
            cy_func(event_func),
            max_allowed(max_allowed_),
            status(CyrkErrorCodes::NO_ERROR),
            direction(direction_),
            initialized(true)
{
    if (this->cy_func)
    {
        this->check = &Event::cy_check;
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

    this->cy_func = event_func;
    if (this->cy_func)
    {
        this->check = &Event::cy_check;
    }
    this->direction   = direction_;
    this->max_allowed = max_allowed_;

    // Everything seems good.
    this->current_count = 0;
    this->last_root   = 0.0;
    this->is_active   = false;
    this->initialized = true;
    this->status = CyrkErrorCodes::NO_ERROR;
    return CyrkErrorCodes::NO_ERROR;
}

double Event::cy_check(
        double t,
        double* y_ptr,
        char* arg_ptr
    ) noexcept
{
    // Wrapper for the provided function.
    return this->cy_func(t, y_ptr, arg_ptr);
}

// Python hooks
double Event::py_check(
        double t,
        double* y_ptr,
        char* arg_ptr
    )
{
    // args are not used with pysolver; any additional args are held by the python class and passed in python land.
    if ((not this->use_pysolver) or (not this->cython_extension_class_instance))
    {
        this->status = CyrkErrorCodes::UNINITIALIZED_CLASS;
        return dbl_NAN;
    }
    return call_pyevent_from_cython(
        this->cython_extension_class_instance,
        this->py_check_method,
        this->pyevent_index,
        t,
        y_ptr
    );
}

CyrkErrorCodes Event::set_cython_extension_instance(
        PyObject* cython_extension_class_instance,
        PyEventMethod pyevent_method)
{
    // Now proceed to installing python functions.
    this->use_pysolver = true;
    if (cython_extension_class_instance) [[likely]]
    {
        this->cython_extension_class_instance = cython_extension_class_instance;
        this->py_check_method                 = pyevent_method;

        // Import the cython/python module (functionality provided by "pysolver_api.h")
        const int import_error = import_CyRK__cy__pysolver_cyhook();
        if (import_error) [[unlikely]]
        {
            this->use_pysolver = false;
            this->status = CyrkErrorCodes::ERROR_IMPORTING_PYTHON_MODULE;
            return this->status;
        }
    }
    
    // Change the function pointer for the event checker to trigger the pyhook version.
    this->check = &Event::py_check;

    return CyrkErrorCodes::NO_ERROR;
}