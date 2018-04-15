import numpy as np
import matplotlib.pyplot as plt
import Optimization
import threading

def fn1(x):
    return 1*(x**4) * np.math.exp(-x)

def q1_g():
    _, axes = plt.subplots(2, 3, figsize=(3*4.5, 2*4)) #figsize=()

    print("Golden Section Method (Local Min)")
    a_g = Optimization.golden_section(fn1, -1, 10, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMin())
    ax = axes[0][0]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(-1, 10))
    print("local min in [-1, 10] : " + str(a_g_iter.getResult()))

    a_g = Optimization.golden_section(fn1, -1, 4.5, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMin())
    ax = axes[0][1]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(-1, 4.5))
    print("local min in [-1, 4.5] : " + str(a_g_iter.getResult()))

    a_g = Optimization.golden_section(fn1, 4.5, 10, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMin())
    ax = axes[0][2]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(4.5, 10))
    print("local min in [4.5, 10] : " + str(a_g_iter.getResult()))

    print("Golden Section Method (Local Max)")
    a_g = Optimization.golden_section(fn1, -1, 10, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMax())
    ax = axes[1][0]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(-1, 10))
    print("local max in [-1, 10] : " + str(a_g_iter.getResult()))

    a_g = Optimization.golden_section(fn1, -1, 4.5, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMax())
    ax = axes[1][1]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(-1, 4.5))
    print("local max in [-1, 4.5] : " + str(a_g_iter.getResult()))

    a_g = Optimization.golden_section(fn1, 4.5, 10, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMax())
    ax = axes[1][2]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(4.5, 10))
    print("local max in [4.5, 10] : " + str(a_g_iter.getResult()))

    plt.tight_layout()

def q1_q():
    _, axes = plt.subplots(2, 3, figsize=(3*4.5, 2*4)) #figsize=()

    print("Quadratic Interpolation Method (Local Min)")
    a_q = Optimization.quadratic_interpolation(fn1, -1, 10, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMin())
    ax = axes[0][0]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(-1, 10))
    print("local min in [-1, 10] : " + str(a_q_iter.getResult()))

    a_q = Optimization.quadratic_interpolation(fn1, -1, 4.5, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMin())
    ax = axes[0][1]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(-1, 4.5))
    print("local min in [-1, 4.5] : " + str(a_q_iter.getResult()))

    a_q = Optimization.quadratic_interpolation(fn1, 4.5, 10, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMin())
    ax = axes[0][2]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(4.5, 10))
    print("local min in [4.5, 10] : " + str(a_q_iter.getResult()))

    print("Quadratic Interpolation Method (Local Max)")
    a_q = Optimization.quadratic_interpolation(fn1, -1, 10, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMax())
    ax = axes[1][0]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(-1, 10))
    print("local max in [-1, 10] : " + str(a_q_iter.getResult()))

    a_q = Optimization.quadratic_interpolation(fn1, -1, 4.5, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMax())
    ax = axes[1][1]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(-1, 4.5))
    print("local max in [-1, 4.5] : " + str(a_q_iter.getResult()))

    a_q = Optimization.quadratic_interpolation(fn1, 4.5, 10, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMax())
    ax = axes[1][2]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(4.5, 10))
    print("local max in [4.5, 10] : " + str(a_q_iter.getResult()))

    plt.tight_layout()

def fn2(x):
    x1 = x[0]
    x2 = x[1]
    return -1*(x1**4 + 4*(x2**2) - x1**2 - 4*x1*x2)

def q2():
    var_range = np.array([[-2, 2], [-2, 2]])
    init_point = np.array([-1, -1])
    a_u = Optimization.univariate(fn2, 2, var_range, init_point, 1e-5)
    y = a_u.findMin()
    plt.plot(y[:, 0].flatten(), y[:, 1].flatten())
    print(y[-1])

if __name__ == '__main__':
    q1_q()
    #q2()
    plt.show()