#include "pysolver_interface.hpp"

// Constructors
PySolverBase::PySolverBase() {}
PySolverBase::PySolverBase(
    // Base Class input arguments
    PyObject* cython_extension_class_instance,
    std::shared_ptr<CySolverResult> storage_ptr,
    const double t_start,
    const double t_end,
    double* y0_ptr,
    size_t num_y,
    bool capture_extra,
    size_t num_extra,
    double* args_ptr,
    size_t max_num_steps,
    size_t max_ram_MB
    ) : CySolverBase(nullptr, storage_ptr, t_start, t_end, y0_ptr, num_y, capture_extra, num_extra, args_ptr, max_num_steps, max_ram_MB) // Pass a nullptr to the base class for the diffeq function since we are overriding it with a call to the cython/python api.
{   

    // Install cython extension instance pointer
    this->cython_extension_class_instance = cython_extension_class_instance;

    // Import the cython/python module (functionality provided by "pysolver_api.h")
    if (import_CyRK__cy__pysolver_cyhook())
    {
        // pass
    }
    else
    {
        Py_XINCREF(this->cython_extension_class_instance);
    }

}

// Deconstructors
PySolverBase::~PySolverBase()
{
    // Decrement the python counter on the extension class instance
    Py_XDECREF(this->cython_extension_class_instance);
}

// Override the diffeq call
void PySolverBase::diffeq()
{
    int diffeq_status;

    if (this->cython_extension_class_instance)
    {
        diffeq_status = call_diffeq_from_cython(this->cython_extension_class_instance);
        if (diffeq_status < 0)
        {
            this->status = -50;
            this->storage_ptr->error_code = -50;
            this->storage_ptr->update_message("Error when calling cython diffeq wrapper from PySolverBase c++ class.\n");
        }
        else
        {
            // TODO: Save results into class instance.
        }
    }
}

PyRK45::PyRK45() {}
PyRK45::PyRK45(
        // Base Class input arguments
        PyObject* cython_extension_class_instance,
        std::shared_ptr<CySolverResult> storage_ptr,
        const double t_start,
        const double t_end,
        double* y0_ptr,
        size_t num_y,
        bool capture_extra,
        size_t num_extra,
        double* args_ptr,
        size_t max_num_steps,
        size_t max_ram_MB,
        // RKSolver input arguments
        double rtol,
        double atol,
        double* rtols_ptr,
        double* atols_ptr,
        double max_step_size,
        double first_step_size) : 
            RK45(nullptr, storage_ptr, t_start, t_end, y0_ptr, num_y, capture_extra, num_extra, args_ptr, max_num_steps, max_ram_MB, rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size)
{   

    // Install cython extension instance pointer
    this->cython_extension_class_instance = cython_extension_class_instance;

    // Import the cython/python module (functionality provided by "pysolver_api.h")
    if (import_CyRK__cy__pysolver_cyhook())
    {
        // pass
    }
    else
    {
        Py_XINCREF(this->cython_extension_class_instance);
    }

}

// Deconstructors
PyRK45::~PyRK45()
{
    // Decrement the python counter on the extension class instance
    Py_XDECREF(this->cython_extension_class_instance);
}

// Override the diffeq call
void PyRK45::diffeq()
{
    int diffeq_status;

    if (this->cython_extension_class_instance)
    {
        diffeq_status = call_diffeq_from_cython(this->cython_extension_class_instance);
        if (diffeq_status < 0)
        {
            this->status = -50;
            this->storage_ptr->error_code = -50;
            this->storage_ptr->update_message("Error when calling cython diffeq wrapper from PySolverBase c++ class.\n");
        }
        else
        {
            // TODO: Save results into class instance.
        }
    }
}