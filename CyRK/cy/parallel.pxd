# Cython declarations for CyRK's C++ batch solver ("c_parallel.hpp"). cimport this to run many independent
# `cysolve_ivp` integrations across C++ threads from your own Cython code; no OpenMP is involved.
from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector

from CyRK.cy.common cimport DiffeqFuncType, PreEvalFunc, JacobianFuncType
from CyRK.cy.events cimport Event
from CyRK.cy.cysolver_api cimport CySolverResult

cdef extern from "c_parallel.cpp" nogil:
    cdef cppclass c_CySolveJob:
        c_CySolveJob()
        # Required
        CySolverResult* solution_ptr
        DiffeqFuncType diffeq_ptr
        double t_start
        double t_end
        vector[double]* y0_vec_ptr
        # Optional (defaults match `cysolve_ivp_noreturn`; a null pointer means "not provided")
        vector[char]* args_vec_ptr
        double rtol
        double atol
        vector[double]* rtols_vec_ptr
        vector[double]* atols_vec_ptr
        size_t expected_size
        size_t num_extra
        size_t max_num_steps
        size_t max_ram_MB
        cpp_bool capture_dense_output
        vector[double]* t_eval_vec_ptr
        PreEvalFunc pre_eval_func
        vector[Event]* events_vec_ptr
        double max_step_size
        double first_step_size
        cpp_bool force_retain_solver
        JacobianFuncType jac_ptr

    size_t c_cysolve_batch(vector[c_CySolveJob]& jobs, size_t num_threads) except +
