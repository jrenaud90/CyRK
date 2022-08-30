import numpy as np
import matplotlib.pyplot as plt

def diffeq(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]

initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8

def diff_plot(t, y, name):
    fig, ax = plt.subplots()
    ax.plot(t, y[0], 'r', label='$y_{0}$')
    ax.plot(t, y[1], 'b', label='$y_{1}$')
    ax.set(xlabel='$t$', ylabel='$y$')
    ax.legend(loc='best')
    
    plt.show()
    fig.savefig(name + '.pdf')
    

if __name__ == '__main__':
    from cyrk import cyrk_ode
    time_domain, y_results, success, message = cyrk_ode(diffeq, time_span, initial_conds, rtol=rtol, atol=atol)
    diff_plot(time_domain, y_results, 'cythonplot')
    print(success, message)