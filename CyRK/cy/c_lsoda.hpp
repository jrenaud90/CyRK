#pragma once

/* LSODA: the Livermore Solver for Ordinary Differential equations with Automatic method
   switching between the non-stiff Adams formulas and the stiff BDF formulas.

   This is a lightly modified copy of the C translation of ODEPACK's LSODA that ships with SciPy
   ("scipy/integrate/src/lsoda.c", public domain / BSD-3-Clause, see LICENSE.md). The changes made
   for CyRK are:
     * The LAPACK dependency (dgetrf / dgetrs / dgbtrf / dgbtrs) was replaced with CyRK's own
       LU routines in "c_lu.hpp" so that CyRK does not have to link against BLAS/LAPACK.
     * The differential equation and Jacobian callbacks take an opaque `user_data` pointer, which
       CyRK uses to route calls back into the owning `CySolverBase` instance.
     * Names exposed outside of this translation unit carry CyRK's "c_" C-level prefix.
     * `bnorm` is now given the band array with the same row offset that the Jacobian was written
       to, matching the original Fortran DPRJA (SciPy's copy drops the offset, which only affects
       the stiffness norm used for method switching).
     * The user's minimum step size is stored in the common block rather than in a local variable.
       In the Fortran, `HMIN` is the common block variable, so SciPy's copy silently discards
       the option.

   References
   ----------
   .. [1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE Solvers", IMACS Transactions
          on Scientific Computation, Vol 1., pp. 55-64, 1983.
   .. [2] L. Petzold, "Automatic selection of methods for solving stiff and nonstiff systems of
          ordinary differential equations", SIAM Journal on Scientific and Statistical Computing,
          Vol. 4, No. 1, pp. 136-148, 1983.
*/

#include <math.h>
#include <string.h>

#include "c_common.hpp"
#include "c_lu.hpp"

/* Struct to hold the LSODA common block variables.

   This serves as a C representation of the Fortran common blocks used in LSODA. The original
   Fortran code type puns doubles and ints in the same common block which makes it impossible to
   decipher which variables are used in which way, so those punned variables are replicated as
   separate variables. While this slightly increases the memory usage it greatly improves code
   clarity and maintainability.
*/
struct c_lsoda_common_t {
    // All double precision variables (combining the double common blocks ls0001 and lsa001).
    double conit, crate, el[13], elco[156], hold, rmax, tesco[36], ccmax, el0, h, hmin, hmxi, hu, rc, tn, uround;
    double tsw, pdest, pdlast, ratio, cm1[12], cm2[5], pdnorm;

    // All integer variables (combining the integer common blocks ls0001 and lsa001).
    int illin, init, lyh, lewt, lacor, lsavf, lwm, liwm, mxstep, mxhnil,
        nhnil, ntrep, nslast, nyh, ialth, ipup, lmax, meo, nqnyh, nslp,
        icf, ierpj, iersl, jcur, jstart, kflag, l, meth, miter, maxord,
        maxcor, msbp, mxncf, n, nq, nst, nfe, nje, nqu, /* lsa001 part */ insufr,
        insufi, ixpr, icount, irflag, jtyp, mused, mxordn, mxords;
};

// Differential equation callback. `neq` is set to -1 by the callback to abort the integration.
typedef void (*c_lsoda_func_t)(int* neq, double* t, double* y, double* ydot, void* user_data);

// Jacobian callback. `pd` is filled in column-major order with `nrowpd` rows. `ml` and `mu` are
// null unless a banded Jacobian was requested.
typedef void (*c_lsoda_jac_t)(int* neq, double* t, double* y, int* ml, int* mu, double* pd, int* nrowpd, void* user_data);

/* Required length of the real work array. LSODA sizes its history array for whichever of the two
   methods needs more room: the Adams formulas reach order 12 and so need 16 entries per equation,
   while the BDF formulas only reach order 5 but must also store the iteration matrix. */

// Length required for a dense Jacobian (jt = 1 or 2).
inline size_t c_lsoda_dense_rwork_size(const size_t num_y) noexcept
{
    const size_t per_equation = (num_y + 9 > 16) ? (num_y + 9) : 16;
    return 22 + num_y * per_equation;
}

// Length required for a banded Jacobian (jt = 4 or 5).
inline size_t c_lsoda_banded_rwork_size(const size_t num_y, const size_t num_lower, const size_t num_upper) noexcept
{
    const size_t banded_per_equation = 2 * num_lower + num_upper + 10;
    const size_t per_equation = (banded_per_equation > 16) ? banded_per_equation : 16;
    return 22 + num_y * per_equation;
}

// Required length of the integer work array.
inline size_t c_lsoda_iwork_size(const size_t num_y) noexcept
{
    return 20 + num_y;
}

void c_lsoda_mark_error(int* istate, int* illin);

void c_lsoda(
    c_lsoda_func_t f,
    int neq,
    double* y,
    double* t,
    double* tout,
    int itol,
    double* rtol,
    double* atol,
    int* itask,
    int* istate,
    int* iopt,
    double* CYRK_RESTRICT rwork,
    int lrw,
    int* CYRK_RESTRICT iwork,
    int liw,
    c_lsoda_jac_t jac,
    const int jt,
    c_lsoda_common_t* S,
    void* user_data
);
