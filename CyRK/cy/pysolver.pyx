# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

# This implementation was modified from:
# https://stackoverflow.com/questions/10126668/can-i-override-a-c-virtual-function-within-python-with-cython

from libcpp cimport bool as cpp_bool, nullptr
cimport cpython.ref as cpy_ref

from CyRK.utils.memory cimport shared_ptr, make_shared


from CyRK.cy.cysolver2 cimport CySolverBase, CySolverResult, PyCySolverResult, RK45

cdef extern from "pysolver_interface.cpp":
    cdef cppclass PySolverBase(CySolverBase):
        PySolverBase()
        PySolverBase(
            cpy_ref.PyObject* cython_extension_class_instance,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            double* y0_ptr,
            size_t num_y,
            cpp_bool capture_extra,
            size_t num_extra,
            double* args_ptr,
            size_t max_num_steps,
            size_t max_ram_MB
        )
        void diffeq()

    cdef cppclass PyRK45(RK45):
        PyRK45()
        PyRK45(
            cpy_ref.PyObject* cython_extension_class_instance,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            double* y0_ptr,
            size_t num_y,
            cpp_bool capture_extra,
            size_t num_extra,
            double* args_ptr,
            size_t max_num_steps,
            size_t max_ram_MB,
            double rtol,
            double atol,
            double* rtols_ptr,
            double* atols_ptr,
            double max_step_size,
            double first_step_size
        )
        void diffeq()



cdef class PySolverTest:

    cdef size_t num_y
    cdef size_t num_dy

    cdef cpp_bool finalized
    cdef PyRK45* cpp_solver_ptr
    cdef PyCySolverResult storage
    cdef shared_ptr[CySolverResult] storage_ptr

    def __cinit__(self, double t_start, double t_end, double[::1] y0_view,
                  size_t num_extra, cpp_bool capture_extra, size_t expected_size,
                  size_t max_num_steps, size_t max_ram_MB):
        self.finalized = False

        # Parse input and prep for building C++ class instances
        self.num_y = len(y0_view)
        cdef double* y0_ptr = &y0_view[0]

        # Create storage on the heap
        self.storage_ptr = make_shared[CySolverResult](self.num_y, num_extra, expected_size)
        self.storage = PyCySolverResult()

        # Set dummy parameters that arn't needed when using the PySolver API
        cdef double* args_ptr = NULL

        self.cpp_solver_ptr = new PyRK45(
            <cpy_ref.PyObject*>self, self.storage_ptr, t_start, t_end, 
            y0_ptr, self.num_y, capture_extra, num_extra, args_ptr, max_num_steps, max_ram_MB,
            1.0e-7, 1.0e-8, NULL, NULL, 100_000_000.0, 0.0
            )
    
    def __dealloc__(self):        
        # Delete C++ solver instance
        if self.cpp_solver_ptr:
            del self.cpp_solver_ptr

    cpdef void finalize(self):
        if not self.finalized:
            self.storage.set_cyresult_pointer(self.storage_ptr)
            self.finalized = True
    
    def diffeq(self):
        
        cdef double* y_now_ptr = self.cpp_solver_ptr.y_now_ptr
        cdef double* dy_now_ptr = self.cpp_solver_ptr.dy_now_ptr

        # TODO: Call the python diffeq function
        # printf("\n\t!!!DEBUG!!!:: diffeq called from within python function <----\n")
        # printf("\n\t\t I can access t_now = %e; len_t = %d\n", t_now, len_t)

        cdef double y0 = y_now_ptr[0]
        cdef double y1 = y_now_ptr[1]

        dy_now_ptr[0] = (1. - 0.01 * y1) * y0
        dy_now_ptr[1] = (0.02 * y0 - 1.) * y1

        return 1
    
    def take_step(self):
        # printf("\n!!!DEBUG!!!:: take step called from within python function <----\n")
        self.cpp_solver_ptr.take_step()

    def check_status(self):
        return self.cpp_solver_ptr.check_status()

    @property
    def success(self):
        if self.finalized:
            return self.storage.success
        
    @property
    def message(self):
        if self.finalized:
            return self.storage.message
    
    @property
    def t(self):
        if self.finalized:
            return self.storage.t
    
    @property
    def y(self):
        if self.finalized:
            return self.storage.y
    
    @property
    def size(self):
        if self.finalized:
            return self.storage.size