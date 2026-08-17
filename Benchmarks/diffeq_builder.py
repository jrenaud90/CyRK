import numpy as np
from numba import njit
import matplotlib.pyplot as plt


def find_diffeq_data(diffeq_name: str):

    if diffeq_name.lower() == 'pendulum':
        save_name = 'pendulum'
        nice_name = "Pendulum"
        time_span = (0., 10.)
        args = (1., 1., 9.81)
        initial_conds = np.asarray((0.01, 0.), dtype=np.float64, order='C')
        diffeq_num = 6
        def diffeq(dy, t, y, l, m, g):

            # External torque
            torque = 0.1 * np.sin(t)

            y0 = y[0]  # Angular deflection [rad]
            y1 = y[1]  # Angular velocity [rad s-1]
            dy[0] = y1
            dy[1] = (-3. * g / (2. * l)) * np.sin(y0) + (3. / (m * l**2)) * torque
        
        def diffeq_scipy(t, y, l, m, g):

            # External torque
            torque = 0.1 * np.sin(t)

            y0 = y[0]  # Angular deflection [rad]
            y1 = y[1]  # Angular velocity [rad s-1]
            dy = np.empty_like(y)
            dy[0] = y1
            dy[1] = (-3. * g / (2. * l)) * np.sin(y0) + (3. / (m * l**2)) * torque
            return dy

        @njit
        def diffeq_njit(dy, t, y, l, m, g):
            # External torque
            torque = 0.1 * np.sin(t)

            y0 = y[0]  # Angular deflection [rad]
            y1 = y[1]  # Angular velocity [rad s-1]
            dy[0] = y1
            dy[1] = (-3. * g / (2. * l)) * np.sin(y0) + (3. / (m * l**2)) * torque
        
        @njit
        def diffeq_njit2(dy, t, y, args):
            # External torque
            torque = 0.1 * np.sin(t)

            y0 = y[0]  # Angular deflection [rad]
            y1 = y[1]  # Angular velocity [rad s-1]
            dy[0] = y1
            dy[1] = (-3. * args[2] / (2. * args[0])) * np.sin(y0) + (3. / (args[1] * args[0]**2)) * torque

        @njit
        def diffeq_scipy_njit(t, y, l, m, g):

            # External torque
            torque = 0.1 * np.sin(t)

            y0 = y[0]  # Angular deflection [rad]
            y1 = y[1]  # Angular velocity [rad s-1]
            dy = np.empty_like(y)
            dy[0] = y1
            dy[1] = (-3. * g / (2. * l)) * np.sin(y0) + (3. / (m * l**2)) * torque
            return dy

        def event1_check(t, y, l, m, g):
            if y[0] > 0.0:
                return 0
            return 1
        
        def event2_check(t, y, l, m, g):
            if y[1] < 0.0:
                return 0
            return 1

        def event3_check(t, y, l, m, g):
            if t > 5.0:
                return 0
            else:
                return 1
        events = (event1_check, event2_check, event3_check)
        events_njit = (njit(event1_check), njit(event2_check), njit(event3_check))
    elif diffeq_name.lower() in ('large_num_y', 'large_num_y_complex'):
        save_name = 'large_numy_complex'
        nice_name = "Many y's (Complex)"
        time_span = (0., 10.)
        args = (-0.5,)
        num_y = 10_000
        initial_conds = 100.0 * np.ones(num_y, dtype=np.float64, order='C')
        diffeq_num = 9
        def diffeq(dy, t, y, decay_rate):

            num_y = 10_000
            decay_rate_use = decay_rate
            
            # This diffeq converges so should be stable
            for i in range(num_y):
                decay_rate_use *= 0.9999
                if i < (num_y - 1):
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0 + y[i + 1]/50.0)
                else:
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0)
        
        def diffeq_scipy(t, y, decay_rate):
            dy = np.empty_like(y)
            num_y = 10_000
            decay_rate_use = decay_rate
            
            # This diffeq converges so should be stable
            for i in range(num_y):
                decay_rate_use *= 0.9999
                if i < (num_y - 1):
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0 + y[i + 1]/50.0)
                else:
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0)
            return dy

        @njit
        def diffeq_njit(dy, t, y, decay_rate):
            num_y = 10_000
            decay_rate_use = decay_rate
            
            # This diffeq converges so should be stable
            for i in range(num_y):
                decay_rate_use *= 0.9999
                if i < (num_y - 1):
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0 + y[i + 1]/50.0)
                else:
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0)
        
        @njit
        def diffeq_njit2(dy, t, y, args):
            num_y = 10_000
            decay_rate_use = args[0]
            
            # This diffeq converges so should be stable
            for i in range(num_y):
                decay_rate_use *= 0.9999
                if i < (num_y - 1):
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0 + y[i + 1]/50.0)
                else:
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0)

        @njit
        def diffeq_scipy_njit(t, y, decay_rate):
            num_y = 10_000
            decay_rate_use = decay_rate
            
            # This diffeq converges so should be stable
            dy = np.zeros_like(y)
            for i in range(num_y):
                decay_rate_use *= 0.9999
                if i < (num_y - 1):
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0 + y[i + 1]/50.0)
                else:
                    dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0)
            return dy

        def event1_check(t, y, decay_rate):
            if y[0] > 50.0:
                return 0
            return 1
        
        def event2_check(t, y, decay_rate):
            if y[1] < 50.0:
                return 0
            return 1

        def event3_check(t, y, decay_rate):
            if t > 5.0:
                return 0
            else:
                return 1
        events = (event1_check, event2_check, event3_check)
        events_njit = (njit(event1_check), njit(event2_check), njit(event3_check))
    elif diffeq_name.lower() == 'large_num_y_simple':
        save_name = 'large_numy_simple'
        nice_name = "Many y's (Simple)"
        time_span = (0., 50.)
        args = tuple()
        num_y = 10_000
        initial_conds = 100.0 * np.ones(num_y, dtype=np.float64, order='C')
        diffeq_num = 10
        def diffeq(dy, t, y):
            dy.fill(0.0)
            dy[0] = np.sin(2.0 * np.pi * t / 10.0)
            
        
        def diffeq_scipy(t, y):
            dy = np.zeros_like(y)
            dy[0] = np.sin(2.0 * np.pi * t / 10.0)
            return dy

        @njit
        def diffeq_njit(dy, t, y):
            dy.fill(0.0)
            dy[0] = np.sin(2.0 * np.pi * t / 10.0)
        
        @njit
        def diffeq_njit2(dy, t, y, args):
            for i in range(num_y):
                dy[i] = 0.0
            dy[0] = np.sin(2.0 * np.pi * t / 10.0)

        @njit
        def diffeq_scipy_njit(t, y):
            dy = np.zeros_like(y)
            dy[0] = np.sin(2.0 * np.pi * t / 10.0)
            return dy

        def event1_check(t, y):
            if y[0] > 50.0:
                return 0
            return 1
        
        def event2_check(t, y):
            if y[1] < 50.0:
                return 0
            return 1

        def event3_check(t, y):
            if t > 5.0:
                return 0
            else:
                return 1
        events = (event1_check, event2_check, event3_check)
        events_njit = (njit(event1_check), njit(event2_check), njit(event3_check))
    elif diffeq_name.lower() in ('predator_prey', 'predprey'):
        save_name = 'predprey'
        nice_name = 'Predator-Prey DiffEq'
        initial_conds = np.asarray((20., 20.), dtype=np.float64)
        args = tuple()
        time_span = (0., 50.)
        diffeq_num = 0
        def diffeq(dy, t, y):
            dy[0] = (1. - 0.01 * y[1]) * y[0]
            dy[1] = (0.02 * y[0] - 1.) * y[1]

        # Create helper function for scipy to work with this kind of diffeq
        def diffeq_scipy(t, y):

            dy = np.zeros_like(y)
            dy[0] = (1. - 0.01 * y[1]) * y[0]
            dy[1] = (0.02 * y[0] - 1.) * y[1]
            return dy
        
        @njit
        def diffeq_njit(dy, t, y):
            dy[0] = (1. - 0.01 * y[1]) * y[0]
            dy[1] = (0.02 * y[0] - 1.) * y[1]
        
        @njit
        def diffeq_njit2(dy, t, y, args):
            dy[0] = (1. - 0.01 * y[1]) * y[0]
            dy[1] = (0.02 * y[0] - 1.) * y[1]
        
        @njit
        def diffeq_scipy_njit(t, y):

            dy = np.zeros_like(y)
            dy[0] = (1. - 0.01 * y[1]) * y[0]
            dy[1] = (0.02 * y[0] - 1.) * y[1]
            return dy

        def event1_check(t, y):
            if y[0] > 50.0:
                return 0
            return 1
        
        def event2_check(t, y):
            if y[1] < 50.0:
                return 0
            return 1

        def event3_check(t, y):
            if t > 5.0:
                return 0
            else:
                return 1
        events = (event1_check, event2_check, event3_check)
        events_njit = (njit(event1_check), njit(event2_check), njit(event3_check))
    elif diffeq_name.lower() in ('robertson', 'stiff'):
        # Robertson chemical kinetics. The three rate constants differ by many orders of magnitude,
        # which makes this very stiff: the explicit methods are held to tiny steps by stability
        # rather than by the error tolerance. This is the problem to use when benchmarking the
        # implicit methods (BDF, LSODA, Radau) against the explicit ones.
        save_name = 'robertson_stiff'
        nice_name = 'Robertson (Stiff)'
        initial_conds = np.asarray((1., 0., 0.), dtype=np.float64, order='C')
        args = (0.04, 1.0e4, 3.0e7)
        time_span = (0., 40.)
        diffeq_num = 11
        def diffeq(dy, t, y, a, b, c):
            y0 = y[0]
            y1 = y[1]
            y2 = y[2]
            dy[0] = -a * y0 + b * y1 * y2
            dy[1] = a * y0 - b * y1 * y2 - c * y1 * y1
            dy[2] = c * y1 * y1

        # Create helper function for scipy to work with this kind of diffeq
        def diffeq_scipy(t, y, a, b, c):
            y0 = y[0]
            y1 = y[1]
            y2 = y[2]
            dy = np.empty_like(y)
            dy[0] = -a * y0 + b * y1 * y2
            dy[1] = a * y0 - b * y1 * y2 - c * y1 * y1
            dy[2] = c * y1 * y1
            return dy

        @njit
        def diffeq_njit(dy, t, y, a, b, c):
            y0 = y[0]
            y1 = y[1]
            y2 = y[2]
            dy[0] = -a * y0 + b * y1 * y2
            dy[1] = a * y0 - b * y1 * y2 - c * y1 * y1
            dy[2] = c * y1 * y1

        @njit
        def diffeq_njit2(dy, t, y, args):
            y0 = y[0]
            y1 = y[1]
            y2 = y[2]
            dy[0] = -args[0] * y0 + args[1] * y1 * y2
            dy[1] = args[0] * y0 - args[1] * y1 * y2 - args[2] * y1 * y1
            dy[2] = args[2] * y1 * y1

        @njit
        def diffeq_scipy_njit(t, y, a, b, c):
            y0 = y[0]
            y1 = y[1]
            y2 = y[2]
            dy = np.empty_like(y)
            dy[0] = -a * y0 + b * y1 * y2
            dy[1] = a * y0 - b * y1 * y2 - c * y1 * y1
            dy[2] = c * y1 * y1
            return dy

        def event1_check(t, y, a, b, c):
            # The first species has decayed by a tenth.
            if y[0] < 0.9:
                return 0
            return 1

        def event2_check(t, y, a, b, c):
            # The third species has built up to a fifth. The three species always sum to one, so
            # this is placed further along the reaction than the first check to make sure the two
            # trigger at different times.
            if y[2] > 0.2:
                return 0
            return 1

        def event3_check(t, y, a, b, c):
            if t > 20.0:
                return 0
            else:
                return 1
        events = (event1_check, event2_check, event3_check)
        events_njit = (njit(event1_check), njit(event2_check), njit(event3_check))
    else:
        raise NotImplementedError("Unknown/Unsupported Benchmark Differential Equation Requested.")

    # Create plotting routine
    if diffeq_name.lower() in ('robertson', 'stiff'):
        def diff_plot(t, y, fig_name=None):
            # The middle species stays around 1e-5 for the whole integration, so it is scaled up to
            # be visible next to the other two. This is how the Robertson problem is usually shown.
            fig, ax = plt.subplots()
            ax.plot(t, y[0], 'r', label='$y_{0}$')
            ax.plot(t, y[1] * 1.0e4, 'b', label='$y_{1} \\times 10^{4}$')
            ax.plot(t, y[2], 'g', label='$y_{2}$')
            ax.set(xlabel='$t$', ylabel='$y$')
            ax.legend(loc='best')

            # Show figure
            plt.show()

            # Save figure
            if fig_name is not None:
                fig.savefig(f'{fig_name}.pdf')
    elif diffeq_name.lower() in ('large_num_y_simple', 'large_num_y', 'large_num_y_complex'):
        def diff_plot(t, y, fig_name=None):
            fig, ax = plt.subplots()
            ax.plot(t, y[0], 'r', label='$y_{0}$')
            ax.plot(t, y[1], 'b', label='$y_{1}$')
            ax.plot(t, y[100], label='$y_{100}$')
            ax.plot(t, y[500], label='$y_{500}$')
            ax.plot(t, y[5000], label='$y_{5000}$')
            ax.plot(t, y[9990], label='$y_{9990}$')
            ax.set(xlabel='$t$', ylabel='$y$')
            ax.legend(loc='best')
            
            # Show figure
            plt.show()
            
            # Save figure
            if fig_name is not None:
                fig.savefig(f'{fig_name}.pdf')
    else:
        def diff_plot(t, y, fig_name=None):
            fig, ax = plt.subplots()
            ax.plot(t, y[0], 'r', label='$y_{0}$')
            ax.plot(t, y[1], 'b', label='$y_{1}$')
            ax.set(xlabel='$t$', ylabel='$y$')
            ax.legend(loc='best')
            
            # Show figure
            plt.show()
            
            # Save figure
            if fig_name is not None:
                fig.savefig(f'{fig_name}.pdf')
    return (
        save_name,
        nice_name,
        initial_conds,
        args,
        time_span,
        diffeq_num,
        diffeq,
        diffeq_scipy,
        diffeq_njit,
        diffeq_njit2,
        diffeq_scipy_njit,
        events,
        events_njit,
        diff_plot
    )