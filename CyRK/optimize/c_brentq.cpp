/* Adapted from the brentq method shipped with SciPy.optimize. 

Originally written by Charles Harris (charles.harris@sdl.usu.edu)

Adapted for CyRK by Joe P. Renaud (joseph.p.renaud@nasa.gov) in August 2025

*/

#include <cmath>

#include "c_brentq.hpp"

/*
  At the top of the loop the situation is the following:

    1. the root is bracketed between xa and xb
    2. xa is the most recent estimate
    3. xp is the previous estimate
    4. |fp| < |fb|

  The order of xa and xp doesn't matter, but assume xp < xb. Then xa lies to
  the right of xp and the assumption is that xa is increasing towards the root.
  In this situation we will attempt quadratic extrapolation as long as the
  condition

  *  |fa| < |fp| < |fb|

  is satisfied. That is, the function value is decreasing as we go along.
  Note the 4 above implies that the right inequlity already holds.

  The first check is that xa is still to the left of the root. If not, xb is
  replaced by xp and the interval reverses, with xb < xa. In this situation
  we will try linear interpolation. That this has happened is signaled by the
  equality xb == xp;

  The second check is that |fa| < |fb|. If this is not the case, we swap
  xa and xb and resort to bisection.

*/

double c_brentq(
        EventFunc func,
        double xa,
        double xb,
        double xtol,
        double rtol,
        size_t iter,
        std::vector<char>& func_data_vec,
        OptimizeInfo* solver_stats,
        CySolverDense* dense_func)
{
    double xpre = xa, xcur = xb;
    double xblk = 0., fblk = 0., spre = 0., scur = 0.;
    double stry, dpre, dblk;
    solver_stats->error_num = CyrkErrorCodes::INITIALIZING;
    char* func_data_ptr = func_data_vec.data();

    // Variables related to dense function
    bool dense_provided = dense_func != nullptr;
    size_t num_dy = 0;
    std::vector<double>& y_vec = solver_stats->y_vec;
    double* y1_ptr = nullptr;
    double* y2_ptr = nullptr;
    if (dense_provided)
    {
        // Resize to the correct y size and set pointers
        num_dy = dense_func->num_dy;
        y_vec.resize(2 * dense_func->num_dy);
        y1_ptr = &y_vec[0];
        y2_ptr = &y_vec[num_dy];

        // By default set the y_ptr to the first set of y values
        solver_stats->y_at_root_ptr = y1_ptr;

        // Call dense function for the two bounds
        dense_func->call(xpre, y1_ptr);
        dense_func->call(xcur, y2_ptr);
    }

    // Call event function for the two bounds.
    double fpre = (*func)(xpre, y1_ptr, func_data_ptr);
    double fcur = (*func)(xcur, y2_ptr, func_data_ptr);

    solver_stats->funcalls = 2;
    if (fpre == 0)
    {
        solver_stats->error_num = CyrkErrorCodes::CONVERGED;
        return xpre;
    }
    if (fcur == 0)
    {
        solver_stats->error_num = CyrkErrorCodes::CONVERGED;
        if (dense_provided)
        {
            // In this case set the y_ptr to the second set of y values
            solver_stats->y_at_root_ptr = y2_ptr;
        }
        return xcur;
    }

    if (signbit(fpre) == signbit(fcur))
    {
        solver_stats->error_num = CyrkErrorCodes::OPTIMIZE_SIGN_ERROR;
        return 0.0;
    }

    solver_stats->iterations = 0;
    for (size_t i = 0; i < iter; i++)
    {
        solver_stats->iterations++;

        if (
                (fpre != 0)
            and (fcur != 0)
            and (signbit(fpre) != signbit(fcur))
           )
        {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }

        if (fabs(fblk) < fabs(fcur))
        {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        // the tolerance is 2*delta
        double delta = (xtol + rtol*fabs(xcur)) / 2.0;
        double sbis  = (xblk - xcur) / 2.0;

        if (
               (fcur == 0)
            or (fabs(sbis) < delta)
           )
        {
            solver_stats->error_num = CyrkErrorCodes::CONVERGED;
            return xcur;
        }

        if (
                (fabs(spre) > delta)
            and (fabs(fcur) < fabs(fpre))
           )
        {
            if (xpre == xblk)
            {
                /* interpolate */
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            }
            else
            {
                /* extrapolate */
                dpre = (fpre - fcur) / (xpre - xcur);
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre*dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2.0 * fabs(stry) < MIN(fabs(spre), 3.0 * fabs(sbis) - delta))
            {
                /* good short step */
                spre = scur;
                scur = stry;
            } else
            {
                /* bisect */
                spre = sbis;
                scur = sbis;
            }
        }
        else
        {
            /* bisect */
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur; fpre = fcur;
        if (fabs(scur) > delta)
        {
            xcur += scur;
        }
        else
        {
            xcur += (sbis > 0 ? delta : -delta);
        }

        if (dense_provided)
        {
            // This function call will also update the data that the y_ptr in solver_stats
            dense_func->call(xcur, y1_ptr);
        }
        fcur = (*func)(xcur, y1_ptr, func_data_ptr);
        solver_stats->funcalls++;
    }
    solver_stats->error_num = CyrkErrorCodes::OPTIMIZE_CONVERGENCE_ERROR;
    return xcur;
}
