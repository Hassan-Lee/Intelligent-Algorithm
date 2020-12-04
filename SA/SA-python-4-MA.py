import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 定义模拟退火父类
class SimulatedAnnealingBase():
    """
    DO SA(Simulated Annealing)

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    x0 : array, shape is n_dim
        initial solution
    T_max :float
        initial temperature
    T_min : float
        end temperature
    L : int
        num of iteration under every temperature（Long of Chain）

    """

    # 父类的初始化函数
    def __init__(self, func, x_init, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        assert T_max > T_min > 0, 'T_max > T_min > 0'
        self.name = 'SimulatedAnnealingBase'
        self.func = func  # 目标函数
        self.T_max = T_max  # 起始温度
        self.T_min = T_min  # 终止温度
        self.L = int(L)  # 每个温度下的最大迭代次数（马尔科夫链长度）
        # 在 best_y 保持不变次数超过 max_stay_counter 时停止
        self.max_stay_counter = max_stay_counter

        self.n_dims = len(x_init)  # func所需要输入的变量个数

        self.best_x = np.array(x_init)  # 初始点
        self.best_y = self.func(self.best_x)  # 初始化 best_y
        self.T = self.T_max  # 初始化当前温度
        self.iter_cycle = 0  # 初始化小循环次数
        self.total_counter = 0  # 初始化总迭代次数
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]  # 初始化每个阶段 best_x, best_y 的记录列表
        self.generation_T = [self.T]

        # 历史结果（供研究退火过程，可弃用）
        self.best_x_history, self.best_y_history = [self.best_x], [self.best_y]
        self.T_history = [self.T]
        self.total_counter_history = [0]

    # 产生新解
    def get_new_x(self, x):
        u = np.random.uniform(-1, 1, size=self.n_dims)  # 产生均匀分布的[Xi,……] u ~ Uniform(0, 1, size = d)
        x_new = x + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return x_new

    # 降温方式
    def cool_down(self):
        self.T = self.T * 0.7  # 降温方法，指数式降温 self.T = self.T_max * exp{0.7}

    # 接受准则
    def judge(self, df):
        if df < 0:  # 新解 < 原解 ---> 直接接受
            return True
        else:
            p = np.exp(-1 * (df / self.T))  # 新解 > 原解 ---> 计算接受概率p
            rand_p = random.random()  # 产生 rand_p ~ Uniform(0, 1)
            if p > rand_p:
                return True
            else:
                return False

    # 辅助进行终止条件判断的函数
    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)  #

    # 主程序
    def run(self):
        x_current, y_current = self.best_x, self.best_y  # 初始化 x_current, y_current
        stay_counter = 0  # 初始化小循环次数
        self.total_counter = 0  # 初始化总迭代次数
        while True:  # 迭代
            for i in range(self.L):  # 等温阶段
                # 产生新解
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                # 接受准则，Metropolis准则
                df = y_new - y_current  # 计算函数值差值
                if self.judge(df):  # 根据接受准则判断是否接受
                    self.total_counter += 1
                    x_current, y_current = x_new, y_new  # 更新 x_current, y_current

                    # 记录退火过程（可弃用）
                    self.total_counter_history.append(self.total_counter)
                    self.best_x_history.append(x_current)
                    self.best_y_history.append(y_current)
                    self.T_history.append(self.T)

                    if y_new < self.best_y:  # 判断是否更新best_x, best_y
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1  # 更新小循环次数
            self.cool_down()  # 降温
            # 更新该等温阶段的 best_x, best_y
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)
            self.generation_T.append(self.T)

            # 两种终止判断条件

            # 1. best_y 长时间不变 ---> 停止迭代
            if len(self.best_y_history) > 1:
                if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                    stay_counter += 1
                else:
                    stay_counter = 0

            # 2. 当前温度 < 最低温度 ---> 停止迭代
            if self.T < self.T_min:
                stop_code = 'Cooled to final temperature'
                break
            if stay_counter > self.max_stay_counter:
                stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break

        print("Number of iteration: ", self.iter_cycle * self.L)
        print(stop_code)
        return self.best_x, self.best_y

    def draw(self):  # 可视化
        if self.best_x.shape[0] == 1:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            x0_list = []
            for i in self.generation_best_X:
                x0_list.append(i[0])
            x_min, x_max = min(x0_list), max(x0_list)
            x = np.arange(x_min, x_max, 0.001)
            y = np.zeros(shape=x.shape)
            for index in range(x.shape[0]):
                y[index] = self.func(np.array([x[index]]))
            plt.plot(x, y)
            for i in range(len(self.generation_best_X)):
                x0, y0, t = self.generation_best_X[i], self.generation_best_Y[i], self.generation_T[i]
                plt.title("当前温度 ： {}".format(str(t) + "\n" + "\n" + "当前解 ： {}".format(str(x0))))
                p = ax.scatter(x0, y0, marker='o')
                # plt.pause(0.01)
                if i != len(self.generation_best_X) - 1:
                    p.remove()

            plt.show()

        elif self.best_x.shape[0] == 2:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x0_list = []
            x1_list = []
            for i in self.generation_best_X:
                x0_list.append(i[0])
                x1_list.append(i[1])
            x_min, x_max = min(x0_list), max(x0_list)
            y_min, y_max = min(x1_list), max(x1_list)
            x = np.arange(x_min, x_max, 0.01)
            y = np.arange(y_min, y_max, 0.01)
            X, Y = np.meshgrid(x, y)
            Z = self.func(np.array([X, Y]))
            ax.plot_surface(X, Y, Z, cmap='turbo', alpha=0.3)

            for i in range(len(self.generation_best_X)):
                x0, x1, y, t = x0_list[i], x1_list[i], self.generation_best_Y[i], self.generation_T[i]
                plt.title("当前温度 ： {}".format(str(t) + "\n" + "\n" + "当前解 ： {}".format(str(x0) + "," + str(x1))))
                p = ax.scatter(x0, x1, y, c='b', marker='o')
                plt.pause(0.01)
                if i != len(self.generation_best_X) - 1:
                    p.remove()

            plt.xlabel("x0")
            plt.ylabel("x1")
            plt.show()


# 定义快速模拟退火子类
class SAFast(SimulatedAnnealingBase):
    '''
    u ~ Uniform(0, 1, size = d)
    y = sgn(u - 0.5) * T * ((1 + 1/T)**abs(2*u - 1) - 1.0)

    xc = y * (upper - lower)
    x_new = x_old + xc

    c = n * exp(-n * quench)
    T_new = T0 * exp(-c * k**quench)
    '''

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(func, x0, T_max, T_min, L, max_stay_counter, **kwargs)
        self.name = 'SAFast'
        self.m, self.n, self.quench = kwargs.get('m', 1), kwargs.get('n', 1), kwargs.get('quench', 1)
        self.lower, self.upper = kwargs.get('lower', -10), kwargs.get('upper', 10)
        self.c = self.m * np.exp(-self.n * self.quench)

    def get_new_x(self, x):
        r = np.random.uniform(-1, 1, size=self.n_dims)
        xc = np.sign(r) * self.T * ((1 + 1.0 / self.T) ** np.abs(r) - 1.0)
        x_new = x + xc * (self.upper - self.lower)
        return x_new

    def cool_down(self):
        self.T = self.T_max * np.exp(-self.c * self.iter_cycle ** self.quench)


# 定义玻尔兹曼模拟退火子类
class SABoltzmann(SimulatedAnnealingBase):
    '''
    std = minimum(sqrt(T) * ones(d), (upper - lower) / (3*learn_rate))
    y ~ Normal(0, std, size = d)
    x_new = x_old + learn_rate * y

    T_new = T0 / log(1 + k)
    '''

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(func, x0, T_max, T_min, L, max_stay_counter, **kwargs)
        self.name = 'SABoltzmann'
        self.lower, self.upper = kwargs.get('lower', -10), kwargs.get('upper', 10)
        self.learn_rate = kwargs.get('learn_rate', 0.5)

    def get_new_x(self, x):
        std = min(np.sqrt(self.T), (self.upper - self.lower) / 3.0 / self.learn_rate) * np.ones(self.n_dims)
        xc = np.random.normal(0, 1.0, size=self.n_dims)
        x_new = x + xc * std * self.learn_rate
        return x_new

    def cool_down(self):
        self.T = self.T * 0.7


# 定义柯西模拟退火子类
class SACauchy(SimulatedAnnealingBase):
    '''
    u ~ Uniform(-pi/2, pi/2, size=d)
    xc = learn_rate * T * tan(u)
    x_new = x_old + xc

    T_new = T0 / (1 + k)
    '''

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(func, x0, T_max, T_min, L, max_stay_counter, **kwargs)
        self.name = 'SACauchy'
        self.learn_rate = kwargs.get('learn_rate', 0.5)

    def get_new_x(self, x):
        u = np.random.uniform(-np.pi / 2, np.pi / 2, size=self.n_dims)
        xc = self.learn_rate * self.T * np.tan(u)
        x_new = x + xc
        # print("x_new:", x_new)
        return x_new

    def cool_down(self):
        self.T = self.T * 0.7
        # print("T:", self.T)


class SALimited(SimulatedAnnealingBase):
    def __init__(self, func, xmin: np.ndarray, xmax: np.ndarray, T_max=1, T_min=1e-6, **kwargs):
        self.xmin, self.xmax = xmin, xmax
        self.x_init = self.xmin + random.random() * (self.xmax - self.xmin)
        super().__init__(func, self.x_init, T_max, T_min, **kwargs)
        self.name = 'SALimited'

    def get_new_x(self, x_old):
        x_new = np.zeros(shape=x_old.shape)
        for i in range(x_old.shape[0]):
            delta = (random.random() - 0.5) * (self.xmax[i] - self.xmin[i]) * np.sqrt(self.T)
            x_new[i] = x_old[i] + delta
            if x_new[i] < (self.xmin[i] + 10e-4) or (10e-4 + x_new[i]) > self.xmax[i]:
                x_new[i] = x_new[i] - 2 * delta
        return x_new

    def cool_down(self):
        self.T = self.T * 0.95

    def run(self):
        # 初始化参数
        x_current = self.x_init
        y_current = self.func(x_current)

        while self.T > self.T_min:
            x_new = self.get_new_x(x_current)
            y_new = self.func(x_new)
            df = y_new - y_current
            if self.judge(df):  # 根据接受准则判断是否接受
                self.total_counter += 1
                x_current, y_current = x_new, y_new  # 更新 x_current, y_current

                # 记录退火过程（可弃用）
                self.total_counter_history.append(self.total_counter)
                self.best_x_history.append(x_current)
                self.best_y_history.append(y_current)
                self.T_history.append(self.T)

                if y_new < self.best_y:  # 判断是否更新best_x, best_y
                    self.best_x, self.best_y = x_new, y_new

                if df > 0:
                    self.cool_down()
                else:
                    self.total_counter += 1

        stop_code = 'Cooled to final temperature'
        print("Number of iteration: ", self.total_counter)
        print(stop_code)
        return self.best_x, self.best_y


# 调试框架
if __name__ == "__main__":

    '''
    --------------------目标函数--------------------
    '''


    def obj_fun(x: np.ndarray):
        return abs(0.2 * x[0]) + 10 * np.sin(5 * x[0]) + 7 * np.cos(4 * x[0])


    def obj_fun2(x: np.ndarray):
        return (1 - x[0]) ** 2 + 100 * (x[0] - x[1] ** 2) ** 2


    demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2


    def Rastrigrin(x: np.ndarray):
        return 20 + x[0]**2 - 10*np.cos(2*np.pi*x[0]) + x[1]**2 - 10*np.cos(2*np.pi*x[1])


    '''
    ---------------模拟退火算法调试框架---------------
    '''

    '''
             ----------创建类的实例----------      
    可供调试的类：
        SimulatedAnnealingBase  一般退火类
        SAFast                  快速退火类
        SABoltzmann             玻尔兹曼退火类
        SACauchy                柯西退火类
    '''

    start = time.time()  # 开始计时
    x_init = np.array([10,10])  # 设置初始点（行向量）
    x_min = np.array([-10, -10])
    x_max = np.array([10, 10])
    demo = SACauchy(Rastrigrin, x_init)

    '''
             ----------单次运行调试----------      
    '''

    print(demo.run())
    demo.draw()
    end = time.time()  # 结束计时
    print("Duration time: %0.3f" % (end - start))  # 打印运行时间

    '''
             ----------调参参考过程----------      
    '''

    # count = 0
    # for i in range(1000):
    #     rel = demo.run()[0]
    #     if abs(rel[0] - (-0.24)) < 0.01:
    #         count += 1
    # print(count)
    # end = time.time()  # 结束计时
    # print("Duration time: %0.3f" % (end - start))  # 打印运行时间

    '''
             ----------类的比较过程----------      
    '''

    # x_init = np.array([10])
    # methods = [SimulatedAnnealingBase, SAFast, SABoltzmann, SACauchy, SALimited]
    # for method in methods:
    #     start = time.time()
    #     if method == SALimited:
    #         xmin = np.array([-10])
    #         xmax = np.array([10])
    #         demo = method(obj_fun, xmin, xmax)
    #     else:
    #         demo = method(obj_fun, x_init)
    #     print("{} is running……".format(demo.name))
    #     best_x, best_y = demo.run()
    #     demo.draw()
    #     print("best_x: {}, best_y: {}".format(best_x, best_y))
    #     end = time.time()  # 结束计时
    #     print("Duration time: %0.3f" % (end - start), end='\n\n')  # 打印运行时间
