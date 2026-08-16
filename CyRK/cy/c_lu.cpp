#include <cmath>
#include <algorithm>

#include "c_lu.hpp"

size_t c_dense_lu_factor(
        const size_t num_rows,
        double* matrix_ptr,
        int* pivot_ptr) noexcept
{
    /* Right-looking unblocked LU with partial pivoting. This mirrors LAPACK's dgetf2; blocking
       is not worth the complexity here because the implicit solvers are dominated by the
       differential equation calls for the problem sizes CyRK targets. */
    size_t singular_row = 0;

    for (size_t k = 0; k < num_rows; k++)
    {
        double* const column_k_ptr = &matrix_ptr[k * num_rows];

        // Find the pivot row (largest magnitude entry at or below the diagonal).
        size_t pivot_row  = k;
        double pivot_size = std::abs(column_k_ptr[k]);
        for (size_t row_i = k + 1; row_i < num_rows; row_i++)
        {
            const double candidate_size = std::abs(column_k_ptr[row_i]);
            if (candidate_size > pivot_size)
            {
                pivot_size = candidate_size;
                pivot_row  = row_i;
            }
        }
        pivot_ptr[k] = (int)pivot_row;

        const double pivot_value = column_k_ptr[pivot_row];
        if (pivot_value == 0.0) [[unlikely]]
        {
            // Matrix is singular. Record the first occurrence but keep going so that the
            // remaining columns are still in a well-defined state.
            if (singular_row == 0)
            {
                singular_row = k + 1;
            }
            continue;
        }

        // Swap the pivot row into place across every column.
        if (pivot_row != k)
        {
            for (size_t col_j = 0; col_j < num_rows; col_j++)
            {
                double* const column_j_ptr = &matrix_ptr[col_j * num_rows];
                std::swap(column_j_ptr[k], column_j_ptr[pivot_row]);
            }
        }

        // Scale the column below the diagonal to build L.
        const double pivot_inverse = 1.0 / pivot_value;
        for (size_t row_i = k + 1; row_i < num_rows; row_i++)
        {
            column_k_ptr[row_i] *= pivot_inverse;
        }

        // Rank-1 update of the trailing submatrix.
        for (size_t col_j = k + 1; col_j < num_rows; col_j++)
        {
            double* const column_j_ptr = &matrix_ptr[col_j * num_rows];
            const double row_k_value   = column_j_ptr[k];
            if (row_k_value != 0.0)
            {
                for (size_t row_i = k + 1; row_i < num_rows; row_i++)
                {
                    column_j_ptr[row_i] -= column_k_ptr[row_i] * row_k_value;
                }
            }
        }
    }

    return singular_row;
}

void c_dense_lu_solve(
        const size_t num_rows,
        const double* lu_ptr,
        const int* pivot_ptr,
        double* rhs_ptr) noexcept
{
    // Forward substitution with the row interchanges applied on the fly (solve L z = P b).
    for (size_t k = 0; k < num_rows; k++)
    {
        const size_t pivot_row = (size_t)pivot_ptr[k];
        if (pivot_row != k)
        {
            std::swap(rhs_ptr[k], rhs_ptr[pivot_row]);
        }

        const double* const column_k_ptr = &lu_ptr[k * num_rows];
        const double z_k = rhs_ptr[k];
        if (z_k != 0.0)
        {
            for (size_t row_i = k + 1; row_i < num_rows; row_i++)
            {
                rhs_ptr[row_i] -= column_k_ptr[row_i] * z_k;
            }
        }
    }

    // Back substitution (solve U x = z).
    for (size_t k = num_rows; k-- > 0; )
    {
        const double* const column_k_ptr = &lu_ptr[k * num_rows];
        rhs_ptr[k] /= column_k_ptr[k];

        const double x_k = rhs_ptr[k];
        if (x_k != 0.0)
        {
            for (size_t row_i = 0; row_i < k; row_i++)
            {
                rhs_ptr[row_i] -= column_k_ptr[row_i] * x_k;
            }
        }
    }
}

size_t c_banded_lu_factor(
        const size_t num_rows,
        const size_t num_lower,
        const size_t num_upper,
        double* band_ptr,
        const size_t band_stride,
        int* pivot_ptr) noexcept
{
    /* Unblocked banded LU with partial pivoting, following LAPACK's dgbtf2. The band array is
       indexed so that entry (row, col) of the original matrix sits at
       `band_ptr[(band_offset + row - col) + col * band_stride]`. Pivoting can push non-zeros up
       to `num_lower` rows higher, which is what the extra leading rows are reserved for. */
    const size_t band_offset = num_lower + num_upper;
    size_t singular_row      = 0;

    // Zero out the fill-in workspace that pivoting may reach into.
    const size_t last_fill_column = std::min(band_offset, num_rows);
    for (size_t col_j = num_upper + 1; col_j < last_fill_column; col_j++)
    {
        double* const column_j_ptr = &band_ptr[col_j * band_stride];
        for (size_t row_i = band_offset - col_j; row_i < num_lower; row_i++)
        {
            column_j_ptr[row_i] = 0.0;
        }
    }

    // Highest column index that currently holds non-zeros in the active row block.
    size_t last_filled_column = 0;

    for (size_t k = 0; k < num_rows; k++)
    {
        double* const column_k_ptr = &band_ptr[k * band_stride];

        // Number of sub-diagonal entries available in this column.
        const size_t num_sub = std::min(num_lower, num_rows - 1 - k);

        // Find the pivot within the band.
        size_t pivot_offset = 0;
        double pivot_size   = std::abs(column_k_ptr[band_offset]);
        for (size_t sub_i = 1; sub_i <= num_sub; sub_i++)
        {
            const double candidate_size = std::abs(column_k_ptr[band_offset + sub_i]);
            if (candidate_size > pivot_size)
            {
                pivot_size   = candidate_size;
                pivot_offset = sub_i;
            }
        }
        pivot_ptr[k] = (int)(k + pivot_offset);

        const double pivot_value = column_k_ptr[band_offset + pivot_offset];
        if (pivot_value == 0.0) [[unlikely]]
        {
            if (singular_row == 0)
            {
                singular_row = k + 1;
            }
            continue;
        }

        last_filled_column = std::max(last_filled_column, std::min(k + num_upper + pivot_offset, num_rows - 1));

        // Swap the pivot row into the diagonal position across the affected columns.
        if (pivot_offset != 0)
        {
            for (size_t col_j = k; col_j <= last_filled_column; col_j++)
            {
                double* const column_j_ptr = &band_ptr[col_j * band_stride];
                std::swap(
                    column_j_ptr[band_offset + k + pivot_offset - col_j],
                    column_j_ptr[band_offset + k - col_j]);
            }
        }

        if (num_sub > 0)
        {
            // Scale the sub-diagonal entries to build L.
            const double pivot_inverse = 1.0 / pivot_value;
            for (size_t sub_i = 1; sub_i <= num_sub; sub_i++)
            {
                column_k_ptr[band_offset + sub_i] *= pivot_inverse;
            }

            // Rank-1 update of the trailing band.
            for (size_t col_j = k + 1; col_j <= last_filled_column; col_j++)
            {
                double* const column_j_ptr = &band_ptr[col_j * band_stride];
                const double row_k_value   = column_j_ptr[band_offset + k - col_j];
                if (row_k_value != 0.0)
                {
                    for (size_t sub_i = 1; sub_i <= num_sub; sub_i++)
                    {
                        column_j_ptr[band_offset + k + sub_i - col_j] -= column_k_ptr[band_offset + sub_i] * row_k_value;
                    }
                }
            }
        }
    }

    return singular_row;
}

void c_banded_lu_solve(
        const size_t num_rows,
        const size_t num_lower,
        const size_t num_upper,
        const double* band_ptr,
        const size_t band_stride,
        const int* pivot_ptr,
        double* rhs_ptr) noexcept
{
    const size_t band_offset = num_lower + num_upper;

    // Forward substitution (solve L z = P b).
    if (num_lower > 0)
    {
        for (size_t k = 0; k + 1 < num_rows; k++)
        {
            const size_t num_sub   = std::min(num_lower, num_rows - 1 - k);
            const size_t pivot_row = (size_t)pivot_ptr[k];
            if (pivot_row != k)
            {
                std::swap(rhs_ptr[k], rhs_ptr[pivot_row]);
            }

            const double* const column_k_ptr = &band_ptr[k * band_stride];
            const double z_k = rhs_ptr[k];
            if (z_k != 0.0)
            {
                for (size_t sub_i = 1; sub_i <= num_sub; sub_i++)
                {
                    rhs_ptr[k + sub_i] -= column_k_ptr[band_offset + sub_i] * z_k;
                }
            }
        }
    }

    // Back substitution (solve U x = z). U has at most `num_lower + num_upper` super-diagonals.
    for (size_t k = num_rows; k-- > 0; )
    {
        const double* const column_k_ptr = &band_ptr[k * band_stride];
        rhs_ptr[k] /= column_k_ptr[band_offset];

        const double x_k = rhs_ptr[k];
        if (x_k != 0.0)
        {
            const size_t num_super = std::min(band_offset, k);
            for (size_t super_i = 1; super_i <= num_super; super_i++)
            {
                rhs_ptr[k - super_i] -= column_k_ptr[band_offset - super_i] * x_k;
            }
        }
    }
}
