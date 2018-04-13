import numpy as np
import matplotlib.pyplot as plt
import Optimization

def fn1(x):
    return 1*(x**4) * np.math.exp(-x)

def fn1_n(x):
    return -1*(x**4) * np.math.exp(-x)

def q1():
    _, axes = plt.subplots(2, 3, figsize=(2*7, 3*2.5)) #figsize=()

    a_g = Optimization.golden_section(fn1, -1, 10, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMax())
    ax = axes[0][0]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(-1, 10))

    a_g = Optimization.golden_section(fn1, -1, 4.5, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMax())
    ax = axes[0][1]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(-1, 4.5))

    a_g = Optimization.golden_section(fn1, 4.5, 10, 1e-5)
    a_g_iter = Optimization.golden_section_iteration(a_g.findMax())
    ax = axes[0][2]
    a_g_iter.plotIteration(ax)
    ax.set_title("Golden Section [{}, {}]".format(4.5, 10))

    a_q = Optimization.quadratic_interpolation(fn1, -1, 10, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMax())
    ax = axes[1][0]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(-1, 10))

    a_q = Optimization.quadratic_interpolation(fn1, -1, 4.5, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMax())
    ax = axes[1][1]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(-1, 4.5))

    a_q = Optimization.quadratic_interpolation(fn1, 4.5, 10, 1e-7)
    a_q_iter = Optimization.quadratic_interpolation_iteration(a_q.findMax())
    #a_q_iter.plotXYIteration()
    ax = axes[1][2]
    a_q_iter.plotIteration(ax)
    ax.set_title("Quadratic interpolation [{}, {}]".format(4.5, 10))

    plt.tight_layout()
    plt.show()

    '''
    print("Golden Section: ")
    print('range :' + str([-1, 10]))
    print('local min : ' + str(a_g['local max/min']))
    print('x : ' + str(a_g['x']))
    printSteps(a_g['iterator'])
    #plotSteps(a_g['iterator'])
    '''

    '''a_q = Optimization.quadratic_interpolation(fn1, -1, 10, 1e-7)
    print("Quadratic Interpolation: ")
    print('range :' + str([-1, 10]))
    print('conv : ' + str(a_q['conv']))
    print('local min : ' + str(a_q['local max/min']))
    print('x : ' + str(a_q['x']))
    printSteps(a_q['iterator'])
    plotQuadraticSteps(a_q['iterator'])

    a_q = Optimization.quadratic_interpolation(fn1_n, -1, 10, 1e-7)
    print("Quadratic Interpolation: ")
    print('range :' + str([-1, 10]))
    print('conv : ' + str(a_q['conv']))
    print('local max : ' + str(-1 * a_q['local max/min']))
    print('x : ' + str(a_q['x']))
    printSteps(a_q['iterator'])
    #plotQuadraticSteps(a_q['iterator'])

    a_q_lh = Optimization.quadratic_interpolation(fn1, -1, 4.5, 1e-7)
    print("Quadratic Interpolation: ")
    print('range :' + str([-1, 4.5]))
    print('conv : ' + str(a_q_lh['conv']))
    print('local min : ' + str(a_q_lh['local max/min']))
    print('x : ' + str(a_q_lh['x']))
    printSteps(a_q_lh['iterator'])
    #plotQuadraticSteps(a_q_lh['iterator'])

    a_q_lh = Optimization.quadratic_interpolation(fn1_n, -1, 4.5, 1e-7)
    print("Quadratic Interpolation: ")
    print('range :' + str([-1, 4.5]))
    print('conv : ' + str(a_q_lh['conv']))
    print('local max : ' + str(-1 * a_q_lh['local max/min']))
    print('x : ' + str(a_q_lh['x']))
    printSteps(a_q_lh['iterator'])

    a_q_rh = Optimization.quadratic_interpolation(fn1, 4.5, 10, 1e-7)
    print("Quadratic Interpolation: ")
    print('range :' + str([4.5, 10]))
    print('conv : ' + str(a_q_rh['conv']))
    print('local min : ' + str(a_q_rh['local max/min']))
    print('x : ' + str(a_q_rh['x']))
    printSteps(a_q_rh['iterator'])

    a_q_rh = Optimization.quadratic_interpolation(fn1_n, 4.5, 10, 1e-7)
    print("Quadratic Interpolation: ")
    print('range :' + str([4.5, 10]))
    print('conv : ' + str(a_q_rh['conv']))
    print('local max : ' + str(-1 * a_q_rh['local max/min']))
    print('x : ' + str(a_q_rh['x']))
    printSteps(a_q_rh['iterator'])'''

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
    plt.show()
    print(y[-1])

if __name__ == '__main__':
    #q1()
    q2()