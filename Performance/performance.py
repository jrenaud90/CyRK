import os.path

import timeit
from datetime import datetime

import numpy as np
import time
from lotkavolterra import (
    lotkavolterra_cy, lotkavolterra_nb, lotkavolterra_args, lotkavolterra_y0,
    lotkavolterra_time_span_1, lotkavolterra_time_span_2)

from pendulum import pendulum_cy, pendulum_nb, pendulum_args, pendulum_y0, pendulum_time_span_1, pendulum_time_span_2

from lorenz import (
    lorenz_cy, lorenz_nb, lorenz_args, lorenz_y0, lorenz_time_span_1, lorenz_time_span_2,
    lorenz_nb_extra, lorenz_cy_extra)

from CyRK.cy.cysolvertest import CySolverPendulum, CySolverLotkavolterra, CySolverLorenzExtra, CySolverLorenz

REPEATS = 4
RTOL = 1.e-6
ATOL = 1.e-8

performance_filename = 'cyrk_performance.csv'
diffeqs = {
    'Lotkavolterra'  : (lotkavolterra_cy, lotkavolterra_nb, lotkavolterra_args, lotkavolterra_y0,
                        (lotkavolterra_time_span_1, lotkavolterra_time_span_2), CySolverLotkavolterra),
    'Pendulum'       : (pendulum_cy, pendulum_nb, pendulum_args, pendulum_y0, (pendulum_time_span_1, pendulum_time_span_2), CySolverPendulum),
    'Lorenz'         : (lorenz_cy, lorenz_nb, lorenz_args, lorenz_y0, (lorenz_time_span_1, lorenz_time_span_2), CySolverLorenz),
    'Lorenz-ExtraOut': (lorenz_cy_extra, lorenz_nb_extra, lorenz_args, lorenz_y0,
                        (lorenz_time_span_1, lorenz_time_span_2), CySolverLorenzExtra)
    }

time_spans = {
    'Small Time': 0,
    'Large Time': 1,
    }

statistics = {
    'cython (avg)': 0,
    'cython (std)': 1,
    'CySolver (avg)': 1,
    'CySolver (std)': 2,
    'numba  (avg)': 3,
    'numba  (std)': 4
    }

integration_methods = {
    'RK23'  : 0,
    'RK45'  : 1,
    'DOP853': 2
    }


def make_performance_file(integration_method_name):

    if integration_method_name not in integration_methods:
        raise ValueError

    performance_filename = f'cyrk_performance-{integration_method_name}.csv'

    # Build Headers
    header_0 = '(all times in ms),'
    header_1 = ','
    header_2 = 'CyRK Version, Run-on Date'

    for d_i, diffeq_name in enumerate(diffeqs):
        for t_i, time_span_name in enumerate(time_spans):
            for s_i, statistics_name in enumerate(statistics):

                if t_i == 0 and s_i == 0:
                    header_0 += f', {diffeq_name}'
                else:
                    header_0 += ','

                if s_i == 0:
                    header_1 += f', {time_span_name}'
                else:
                    header_1 += ','

                header_2 += f', {statistics_name}'

    with open(performance_filename, 'x') as performance_file:
        performance_file.write(header_0 + '\n')
        performance_file.write(header_1 + '\n')
        performance_file.write(header_2 + '\n')

    print('CyRK Performance File Created.')


def run_performance(integration_method_name):

    if integration_method_name not in integration_methods:
        raise ValueError
    int_method = integration_methods[integration_method_name]
    performance_filename = f'cyrk_performance-{integration_method_name}.csv'

    from CyRK import __version__, cyrk_ode, nbrk_ode
    print(f'Running Performance for CyRK v{__version__} and {integration_method_name}.')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    performance_csv_line = f'{__version__}, {dt_string}'

    # Run performance checks
    for d_i, diffeq_name in enumerate(diffeqs):

        print(f'\tWorking on {diffeq_name}')
        cy_diffeq, nb_diffeq, args_, y0, timespans, CySolverClass = diffeqs[diffeq_name]

        for t_i, time_span_name in enumerate(time_spans):

            print(f'\t\t{time_span_name}')
            t_index = time_spans[time_span_name]
            time_span = timespans[t_index]

            if 'extraout' in diffeq_name.lower():
                cy_timer = timeit.Timer(
                    lambda: cyrk_ode(cy_diffeq, time_span, y0, args=args_, rtol=RTOL, atol=ATOL, rk_method=int_method,
                                     capture_extra=True, num_extra=3))
                nb_timer = timeit.Timer(
                    lambda: nbrk_ode(nb_diffeq, time_span, y0, args=args_, rtol=RTOL, atol=ATOL, rk_method=int_method,
                                     capture_extra=True))
                cysolver_timer = timeit.Timer(
                    lambda: CySolverClass(time_span, y0, args=args_, rtol=RTOL, atol=ATOL, rk_method=int_method,
                                          capture_extra=True, num_extra=3, auto_solve=True))
            else:
                cy_timer = timeit.Timer(lambda: cyrk_ode(cy_diffeq, time_span, y0, args=args_, rtol=RTOL, atol=ATOL,
                                                         rk_method=int_method))
                nb_timer = timeit.Timer(lambda: nbrk_ode(nb_diffeq, time_span, y0, args=args_, rtol=RTOL, atol=ATOL,
                                                         rk_method=int_method))
                cysolver_timer = timeit.Timer(lambda: CySolverClass(time_span, y0, args=args_, rtol=RTOL, atol=ATOL,
                                                                    rk_method=int_method, auto_solve=True))

            # Run the numba function once to make sure everything is compiled.
            print('\t\tPrecompiling numba')
            _ = nbrk_ode(nb_diffeq, time_span, y0, args_, rtol=RTOL, atol=ATOL, rk_method=int_method)

            # Cython
            print('\t\t\tWorking on cyrk_ode.', end='')
            cython_times = list()
            time_0 = time.time()
            for i in range(REPEATS):
                N, T = cy_timer.autorange()
                cython_times.append(T / N * 1000.)
            print(f' Finished taking {time.time() - time_0:0.1f}s.')
            cython_times = np.asarray(cython_times)

            # Store cython results
            cy_avg = np.average(cython_times)
            cy_std = np.std(cython_times)
            performance_csv_line += f', {cy_avg:0.4f}, {cy_std:0.4f}'

            # Cython Solver Class
            print('\t\t\tWorking on CySolver.', end='')
            cysolver_times = list()
            time_0 = time.time()
            for i in range(REPEATS):
                N, T = cysolver_timer.autorange()
                cysolver_times.append(T / N * 1000.)
            print(f' Finished taking {time.time() - time_0:0.1f}s.')
            cysolver_times = np.asarray(cysolver_times)

            # Store Cython Solver results
            cysolver_avg = np.average(cysolver_times)
            cysolver_std = np.std(cysolver_times)
            performance_csv_line += f', {cysolver_avg:0.4f}, {cysolver_std:0.4f}'

            # Numba
            print('\t\t\tWorking on nbrk_ode.', end='')
            numba_times = list()
            time_0 = time.time()
            for i in range(REPEATS):
                N, T = nb_timer.autorange()
                numba_times.append(T / N * 1000.)
            print(f' Finished taking {time.time() - time_0:0.1f}s.')
            numba_times = np.asarray(numba_times)

            # Store numba results
            nb_avg = np.average(numba_times)
            nb_std = np.std(numba_times)
            performance_csv_line += f', {nb_avg:0.4f}, {nb_std:0.4f}'

    # Save results to disk
    if not os.path.isfile(performance_filename):
        print('CyRK Performance File Not Found. Creating...')
        make_performance_file(integration_method_name)

    with open(performance_filename, 'a') as performance_file:
        performance_file.write(performance_csv_line + '\n')


if __name__ == '__main__':

    for integration_method_name in integration_methods:
        performance_filename = f'cyrk_performance-{integration_method_name}.csv'

        # Check if performance file exists. Create it if it does not.
        if not os.path.isfile(performance_filename):
            print('CyRK Performance File Not Found. Creating...')
            make_performance_file(integration_method_name)

        run_performance(integration_method_name)
