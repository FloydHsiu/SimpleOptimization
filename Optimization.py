import numpy as np
import matplotlib.pyplot as plt

'''Single variable unconstraint optimization'''
class golden_section:
    def __init__(self, func, a, b, tol):
        self.func = func
        self.a = a
        self.b = b
        self.tol = tol
    
    def start(self, func, ismin):
        d_iterator = []
        alpha = (np.sqrt(5) - 1) / 2
        a = self.a
        b = self.b
        if a > b:
            tmp = a
            a = b
            b = tmp
        l = a + (1 - alpha)*(b - a)
        r = b - (1 - alpha)*(b - a)
        yl = func(l)
        yr = func(r)
        #keep iterator datas
        d = {'a':a, 'b':b, 'l':l, 'r':r, 'yl':yl, 'yr':yr}
        d_iterator.append(d)
        while (b - a) > self.tol:
            if yl > yr:
                a = l
                l = r
                r = b - (1 - alpha)*(b - a)
                yl = yr
                yr = func(r)
            else:
                b = r
                r = l
                l = a + (1 - alpha)*(b - a)
                yr = yl
                yl = func(l)
            #keep iterator datas
            if ismin:
                d = {'a':a, 'b':b, 'l':l, 'r':r, 'yl':yl, 'yr':yr}
            else:
                d = {'a':a, 'b':b, 'l':l, 'r':r, 'yl':-1*yl, 'yr':-1*yr}
            d_iterator.append(d)
        result = {}
        if yl < yr:
            if ismin:
                result['localmin'] = yl
            else:
                result['localmin'] = yl * -1
            result['x'] = l
        else:
            if ismin:
                result['localmin'] = yr
            else:
                result['localmin'] = yr * -1
            result['x'] = r
        result['iteration'] = d_iterator
        return result

    def findMin(self):
        return self.start(self.func, True)
    
    def findMax(self):
        funcMax = lambda x: self.func(x) * -1
        return self.start(funcMax, False)

class golden_section_iteration:
    def __init__(self, result):
        self.localmin = result['localmin']
        self.x = result['x']
        self.iteration = result['iteration']
        self.times = len(self.iteration)

    def plotIteration(self, ax):
        x = np.linspace(0, self.times-1, self.times)
        yl = np.array([])
        yr = np.array([])
        for d in self.iteration:
            yl = np.append(yl, d['yl'])
            yr = np.append(yr, d['yr'])

        ax.plot(x, yl, marker='.', linewidth=1, label='f(l)')
        ax.plot(x, yr, marker='.', linewidth=1, label='f(r)')
        ax.set_xlabel('Iteration times')
        ax.set_ylabel('Func(x)')
        ax.set_aspect('auto')

        ax.legend(loc='best')
        return ax

    def printIteration(self):
        print('\n' + 'iteration step:')
        for i in range(0, self.times):
            print('iterate {:^3d}: '.format(i) + str(self.iteration[i]))
        print('\n')

    def getResult(self):
        result = {}
        result["x"] = self.x
        result["f(x)"] = self.localmin
        return result

class quadratic_interpolation:
    def __init__(self, func, l, u, tol):
        self.func = func
        self.l = l
        self.u = u
        self.tol = tol

    def cal_epsilon(self, fval, qval):
        if fval != 0:
            return np.abs((fval - qval)/fval)
        elif fval == 0 and qval != 0:
            return np.abs((fval - qval)/qval)
        else:
            return 0
    
    def start(self, func, ismin):
        d_iterator = []
        l = self.l
        u = self.u
        tol = self.tol
        if l > u:
            tmp = l
            l = u
            u = tmp
        m = (l+u)/2
        #functions
        fn_f = func
        fn_q = lambda x: fl*(x-m)*(x-u)/((l-m)*(l-u)) + fm*(x-l)*(x-u)/((m-l)*(m-u)) +\
            fu*(x-l)*(x-m)/((u-l)*(u-m))
        cal_x_star = lambda : 0.5 * (fl*(m**2-u**2) + fm*(u**2-l**2) + fu*(l**2-m**2))/\
            (fl*(m-u) + fm*(u-l) + fu*(l-m))
        #cal_epsilon = lambda : np.abs(((fn_f(x_star) - fn_q(x_star))/fn_f(x_star)))
        
        #cal l, m, u, x_star fn value
        fl, fm, fu = fn_f(l), fn_f(m), fn_f(u)
        x_star = cal_x_star()
        if x_star < l: x_star = l
        elif x_star > u: x_star = u
        fx_star = fn_f(x_star)
        #keep iterator datas
        d = {'l': l, 'f(l)': fl, 'm':m, 'f(m)': fm, 'u':u, 'f(u)':fu, 'x*':x_star, 'f(x*)':fx_star}
        d_iterator.append(d)

        while self.cal_epsilon(fn_f(x_star), fn_q(x_star)) > tol:
            if x_star <= m:
                if fm >= fx_star:
                    u = m
                    fu = fm
                    m = x_star
                    fm = fx_star
                else:
                    l = x_star
                    fl = fx_star
            else:
                if fm >= fx_star:
                    l = m
                    fl = fm
                    m = x_star
                    fm = fx_star
                else:
                    u = x_star
                    fu = fx_star
            #refresh x_star fn value
            #fl, fm, fu = fn_f(l), fn_f(m), fn_f(u)
            x_star = cal_x_star()
            fx_star = fn_f(x_star)
            #keep iterator datas
            if ismin: 
                d = {'l': l, 'f(l)': fl, 'm':m, 'f(m)': fm, 'u':u, 'f(u)':fu, 'x*':x_star, 'f(x*)':fx_star}
            else:
                d = {'l': l, 'f(l)': -1*fl, 'm':m, 'f(m)': -1*fm, 'u':u, 'f(u)':-1*fu, 'x*':x_star, 'f(x*)':-1*fx_star}
            d_iterator.append(d)
            #not converg
            if len(d_iterator) > 50:
                break

        result = {}
        if ismin:
            result['localmin'] = fx_star
        else:
            result['localmin'] = -1*fx_star
        result['x'] = x_star
        result['iteration'] = d_iterator
        if self.cal_epsilon(fn_f(x_star), fn_q(x_star)) > tol : result['conv'] = False
        else: result['conv'] = True
        return result

    def findMin(self):
        return self.start(self.func, True)

    def findMax(self):
        funcMax = lambda x: self.func(x) * -1
        return self.start(funcMax, False)

class quadratic_interpolation_iteration:
    def __init__(self, result):
        self.localmin = result['localmin']
        self.x = result['x']
        self.iteration = result['iteration']
        self.conv = result['conv']
        self.times = len(self.iteration)

    def plotIteration(self, ax):
        x = np.linspace(0, self.times-1, self.times)

        yl = np.array([])
        ym = np.array([])
        yu = np.array([])
        ystar = np.array([])
        for d in self.iteration:
            yl = np.append(yl, d['f(l)'])
            ym = np.append(ym, d['f(m)'])
            yu = np.append(yu, d['f(u)'])
            ystar = np.append(ystar, d['f(x*)'])

        #ax.plot(x, yl, linewidth=1, label='f(l)')
        #ax.plot(x, ym, linewidth=1, label='f(m)')
        #ax.plot(x, yu, linewidth=1, label='f(u)')
        ax.plot(x, ystar, marker='.', linewidth=1, label='f(x*)')
        ax.set_xlabel('Iteration times')
        ax.set_ylabel('Func(x)')
        ax.set_aspect('auto')

        ax.legend(loc='best')
        return ax

    def plotXIteration(self):
        x = np.linspace(0, self.times-1, self.times)

        _, ax = plt.subplots()

        xl = np.array([])
        xm = np.array([])
        xu = np.array([])
        xstar = np.array([])
        for d in self.iteration:
            xl = np.append(xl, d['l'])
            xm = np.append(xm, d['m'])
            xu = np.append(xu, d['u'])
            xstar = np.append(xstar, d['x*'])

        ax.plot(xl, x, linewidth=1, label='f(l)')
        ax.plot(xm, x, linewidth=1, label='f(m)')
        ax.plot(xu, x, linewidth=1, label='f(u)')
        ax.plot(xstar, x, linewidth=1, label='f(x*)')
        ax.set_xlabel('X')
        ax.set_ylabel('Iteration times')
        ax.set_aspect('auto')

        ax.legend(loc='best')
        return ax

    def plotXYIteration(self):
        _, ax = plt.subplots()

        xl = np.array([])
        xm = np.array([])
        xu = np.array([])
        xstar = np.array([])
        yl = np.array([])
        ym = np.array([])
        yu = np.array([])
        ystar = np.array([])
        for d in self.iteration:
            xl = np.append(xl, d['l'])
            xm = np.append(xm, d['m'])
            xu = np.append(xu, d['u'])
            xstar = np.append(xstar, d['x*'])
            yl = np.append(yl, d['f(l)'])
            ym = np.append(ym, d['f(m)'])
            yu = np.append(yu, d['f(u)'])
            ystar = np.append(ystar, d['f(x*)'])

        #ax.plot(xl, x, linewidth=1, label='f(l)')
        #ax.plot(xm, x, linewidth=1, label='f(m)')
        #ax.plot(xu, x, linewidth=1, label='f(u)')
        ax.plot(xstar, ystar, marker='o', linewidth=1, label='x*')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('auto')

        ax.legend(loc='best')
        return ax
    
    def plotQuadraticFunc(self, times):
        x = np.linspace(-1, 10, 1100)
        _, ax = plt.subplots()
        for t in range(0, times):
            if t < self.times:
                iterate = self.iteration[t]
                fn_q = lambda x: fl*(x-m)*(x-u)/((l-m)*(l-u)) + fm*(x-l)*(x-u)/((m-l)*(m-u)) +\
                                fu*(x-l)*(x-m)/((u-l)*(u-m))
                l = iterate['l']
                m = iterate['m']
                u = iterate['u']
                fl = iterate['f(l)']
                fm = iterate['f(m)']
                fu = iterate['f(u)']
                star = iterate['x*']
                fstar = iterate['f(x*)']
                y = np.array([])
                for i in range(0, len(x)):
                    y = np.append(y, fn_q(x[i]))
                ax.plot(x, y, label='{}'.format(t))
                ax.plot(star, fstar, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('auto')
        ax.legend(loc='best')

    def printIteration(self):
        print('\n' + 'iteration step:')
        for i in range(0, self.times):
            print('iterate {:^3d}: '.format(i) + str(self.iteration))
        print('\n')

    def getResult(self):
        result = {}
        result["x"] = self.x
        result["f(x)"] = self.localmin
        return result

class univariate:
    def __init__(self, func, var_size, var_range, init_point, tol):
        if var_size == len(var_range):
            Exception()
        self.func = func
        self.var_size = var_size
        self.var_range = var_range
        self.init_point = init_point
        self.tol = tol
        self.directions = 0.1 * np.identity(self.var_size)

    def func_lambda(self, func, y, d):
        func_l = lambda x: func(y+x*d)
        return func_l

    def start(self, ismin):
        k = 0
        times = 0
        #step1
        y = np.array([self.init_point])
        a_g = golden_section(self.func_lambda(self.func, y[0], self.directions[0]), \
                            self.var_range[0][0], self.var_range[0][1], 10-5)
        if(ismin): a_g_iter = golden_section_iteration(a_g.findMin())
        else: a_g_iter = golden_section_iteration(a_g.findMax())
        min_lambda = a_g_iter.getResult()["x"]
        #dbg = np.array([y[0]+min_lambda*self.directions[0]])
        y = np.append(y, np.array([y[0]+min_lambda*self.directions[0]]), axis=0)
        for i in range(1, self.var_size):
            a_g = golden_section(self.func_lambda(self.func, y[i], self.directions[i]), \
                            self.var_range[i][0], self.var_range[i][1], 10-5)
            if(ismin): a_g_iter = golden_section_iteration(a_g.findMin())
            else: a_g_iter = golden_section_iteration(a_g.findMax())
            min_lambda = a_g_iter.getResult()["x"]
            y = np.append(y, np.array([y[i]+min_lambda*self.directions[i]]), axis=0)
        k = self.var_size
        times = times+1
        #step2
        while np.linalg.norm(y[self.var_size] - y[self.var_size-1]) > self.tol:
            for i in range(0, self.var_size):
                a_g = golden_section(self.func_lambda(self.func, y[k+i], self.directions[i]), \
                                self.var_range[i][0], self.var_range[i][1], 10-7)
                if(ismin): a_g_iter = golden_section_iteration(a_g.findMin())
                else: a_g_iter = golden_section_iteration(a_g.findMax())
                min_lambda = a_g_iter.getResult()["x"]
                y = np.append(y, np.array([y[k+i]+min_lambda*self.directions[i]]), axis=0)
            k = self.var_size + k
            times = times+1
            if times > 1000:
                break

        return y

    def findMin(self):
        return self.start(True)

    def findMax(self):
        return self.start(False)

'''Multi-variable unconstraint optimization'''

class downhill_simplex:
    alpha = 1
    gamma = 2
    beta = 0.5
    def __init__(self, func, vecs, tol):
        self.func = func
        self.vecs = np.array(vecs)
        self.tol = tol
        self.x = {}
        self.x['max'] = vecs[0]
        self.x['max2'] = vecs[0]
        self.x['min'] = vecs[0]
        self.fn_x = {}
        self.max_index = 0
        return

    def find_x_minmax(self):
        fmax = self.func(self.x['max'])
        fmax2 = self.func(self.x['max2'])
        fmin = self.func(self.x['min'])
        for i in range(0, len(self.vecs)):
            fval = self.func(self.vecs[i])
            if fval > fmax:
                self.x['max2'] = self.x['max']
                self.x['max'] = self.vecs[i]
                fmax = self.func(self.x['max'])
                fmax2 = self.func(self.x['max2'])
                self.max_index = i
            elif fval > fmax2:
                self.x['max2'] = self.vecs[i]
                fmax2 = self.func(self.x['max2'])
            else:pass
            if fval < fmin:
                self.x['min'] = self.vecs[i]
                fmin = self.func(self.x['min'])
        self.fn_x['max'] = fmax
        self.fn_x['max2'] = fmax2
        self.fn_x['min'] = fmin
        return

    def stop_critarial(self):
        if abs(self.x['max'] - self.x['min']) < self.tol:
            return True
        else:
            return False

    def cal_x_average(self):
        return (self.vecs.sum(axis=0) - self.x['max']) / (len(self.vecs) - 1)

    def cal_x_reflect(self):
        return self.x['max'] + self.alpha * (self.x['average'] - self.x['max'])

    def cal_x_expansion(self):
        return self.x['average'] + self.gamma * (self.x['reflect'] - self.x['average'])
    
    def start(self):
        #step1
        self.find_x_minmax()
        while not self.stop_critarial():
            self.x['average'] = self.cal_x_average()
            #step2
            self.x['reflect'] = self.cal_x_reflect()
            self.fn_x['reflect'] = self.func(self.x['reflect'])
            if self.fn_x['min'] > self.fn_x['reflect']:
                self.x['expansion'] = self.cal_x_expansion()
                #step3
                self.fn_x['expansion'] = self.func(self.x['expansion'])
                if self.fn_x['reflect'] > self.fn_x['expansion']:
                    self.vecs[self.max_index] = self.x['expansion']
                else:
                    self.vecs[self.max_index] = self.x['reflect']
                #goto step1
            else:
                #step4
                if self.fn_x['max2'] >= self.fn_x['reflect']:
                    self.vecs[self.max_index] = self.x['reflect']#goto step1
                else:
                    #step5
                    if self.fn_x['reflect'] - self.fn_x['max'] > 0: 
                        self.x['p'] = self.x['max']
                        self.fn_x['p'] = self.fn_x['max']
                    else: 
                        self.x['p'] = self.x['reflect']
                        self.fn_x['p'] = self.fn_x['reflect']
                    self.x['contraction'] = self.x['average'] + self.beta * (self.x['p'] - self.x['average'])
                    self.fn_x['contraction'] = self.func(self.x['contraction'])
                    if self.fn_x['contraction'] > self.fn_x['p']:
                        for i in range(0, len(self.vecs)):
                            self.vecs[i] = self.vecs[i] + (self.x['min'] - self.vecs[i]) / 2
                        #goto step1
                    else:
                        self.vecs[self.max_index] = self.x['contraction']#goto step 1
            #step1
            self.find_x_minmax()
        return