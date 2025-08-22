# Optimize Package

## BrentQ Method
This function utilizes the Brent's method to find a root of the function on the sign changing interval [a, b]. It is based off of the implementation used by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html).

### Limitations
While this function is wrapped in Cython it is not currently exposed to Python so you must use Cython or C++ to access and use it (see examples below). 

### Example Usage
```cython
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libcpp.vector cimport vector

from CyRK.optimize.brentq cimport c_brentq, OptimizeInfo

# extra parameters
ctypedef struct extra_params:
    double[4] a

# callback function
cdef double f_example(double x, char *args_char_ptr) noexcept nogil:
    cdef extra_params *args_ptr = <extra_params*> args_char_ptr
    # use Horner's method
    return ((args_ptr.a[3]*x + args_ptr.a[2])*x + args_ptr.a[1])*x + args_ptr.a[0]

def brentq_test(
        double a0,
        tuple args,
        double xa,
        double xb,
        double xtol,
        double rtol,
        size_t mitr):
    
    cdef vector[char] args_vec = vector[char](sizeof(extra_params))
    cdef extra_params* args_ptr = <extra_params*>args_vec.data()
    cdef OptimizeInfo optimize_info

    cdef size_t i
    for i in range(4):
        if i == 0:
            args_ptr.a[i] = a0
        else:
            args_ptr.a[i] = args[i-1]

    return c_brentq(f_example, xa, xb, xtol, rtol, mitr, args_vec, &optimize_info)
```

### Citation
Use of this function or of CyRK's integrators with "events" on should cite [SciPy](https://scipy.org/citing-scipy/) since this implementation is based on their work.