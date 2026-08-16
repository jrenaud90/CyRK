#pragma once

#include <complex>
#include <cstddef>

/* Dense and banded LU factorizations used by CyRK's implicit integrators.

   These routines cover the small subset of LAPACK (dgetrf / dgetrs / dgbtrf / dgbtrs) that the
   BDF and LSODA solvers require. They are implemented here so that CyRK does not need to link
   against an external BLAS/LAPACK library.

   All matrices use LAPACK's column-major storage: entry (row, col) of a dense `num_rows` square
   matrix lives at `matrix_ptr[row + col * num_rows]`.

   Banded matrices use LAPACK's band storage with room for fill-in: entry (row, col) of the
   original matrix lives at `band_ptr[(num_lower + num_upper + row - col) + col * band_stride]`
   where `band_stride >= 2 * num_lower + num_upper + 1`. The first `num_lower` rows of the array
   are workspace that the factorization fills in.
*/

// Factor a dense square matrix in place using partial pivoting (equivalent to LAPACK's dgetrf).
// `pivot_ptr` must hold at least `num_rows` entries; on exit `pivot_ptr[k]` is the zero-based
// index of the row that was swapped with row k.
// Returns 0 on success or the one-based index of the first exactly-zero pivot.
size_t c_dense_lu_factor(
    const size_t num_rows,
    double* matrix_ptr,
    int* pivot_ptr) noexcept;

// Solve A x = b for a single right-hand side using the output of `c_dense_lu_factor`
// (equivalent to LAPACK's dgetrs). `rhs_ptr` holds b on entry and x on exit.
void c_dense_lu_solve(
    const size_t num_rows,
    const double* lu_ptr,
    const int* pivot_ptr,
    double* rhs_ptr) noexcept;

// Complex versions of the two routines above (equivalent to LAPACK's zgetrf and zgetrs). The
// Radau integrator needs these because it factorizes a complex iteration matrix.
size_t c_complex_dense_lu_factor(
    const size_t num_rows,
    std::complex<double>* matrix_ptr,
    int* pivot_ptr) noexcept;

void c_complex_dense_lu_solve(
    const size_t num_rows,
    const std::complex<double>* lu_ptr,
    const int* pivot_ptr,
    std::complex<double>* rhs_ptr) noexcept;

// Factor a banded square matrix in place using partial pivoting (equivalent to LAPACK's dgbtrf).
// Returns 0 on success or the one-based index of the first exactly-zero pivot.
size_t c_banded_lu_factor(
    const size_t num_rows,
    const size_t num_lower,
    const size_t num_upper,
    double* band_ptr,
    const size_t band_stride,
    int* pivot_ptr) noexcept;

// Solve A x = b for a single right-hand side using the output of `c_banded_lu_factor`
// (equivalent to LAPACK's dgbtrs). `rhs_ptr` holds b on entry and x on exit.
void c_banded_lu_solve(
    const size_t num_rows,
    const size_t num_lower,
    const size_t num_upper,
    const double* band_ptr,
    const size_t band_stride,
    const int* pivot_ptr,
    double* rhs_ptr) noexcept;
