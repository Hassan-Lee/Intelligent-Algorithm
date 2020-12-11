
import numpy as np
import random
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 二维目标函数

def Rastrigrin(x: np.ndarray):
    return 20 + x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1])

def obj_fun(x: np.ndarray):
    return abs(0.2 * x[0]) + 10 * np.sin(5 * x[0]) + 7 * np.cos(4 * x[0])


def obj_fun2(x: np.ndarray):
    return (1 - x[0]) ** 2 + 100 * (x[0] - x[1] ** 2) ** 2


def Easom(x: np.ndarray):
    return -1 * np.cos(x[0]) * np.cos(x[1]) * np.exp(-1 * (x[0] - np.pi) ** 2 - (x[0] - np.pi) ** 2)


def obj_fun_test1(x: np.ndarray):
    return -1 * (10 + np.sin(1 / x) / ((x - 0.16) ** 2 + 0.1))


def Bohachevsky(x: np.ndarray):
    return x[0] ** 2 + x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) + 0.3 * np.cos(4 * np.pi * x[1]) + 0.3



def Schaffersf6(x: np.ndarray):
    return 0.5 + (np.sin(x[0] ** 2 + x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2) ** 2


def Shubert(x: np.ndarray):
    s1 = 0
    s2 = 0
    for i in range(1, 6):
        s1 += i * np.cos((i + 1) * x[0] + i)
        s2 += i * np.cos((i + 1) * x[1] + i)
    return s1 + s2

# 一维目标函数
def fun1(x):
    return -(x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x))

class GSA():
    def __init__(self,T_max=400,T_min=10,pop=10,new_pop=10,cur_g=1,p=0.9,tour_n=10,func=fun1,shape=1, **kwargs):
        """
        @param T_max: 最大温度
        @param T_min: 最小温度
        @param pop: 种群的个体数量
        @param new_pop: 每个个体生成的新解的数量
        @param cur_g: 当前迭代代数
        @param p: 锦标赛中的概率p
        @param tour_n: 一次锦标赛的选手数
        @param func: 函数
        @param x_best: 最优点
        @param y_best: 最优点的函数值
        @param shape: x的维度
        """
        self.T_max = T_max
        self.T = T_max  # 当前温度
        self.T_min = T_min
        self.pop = pop
        self.new_pop = new_pop
        self.cur_g = cur_g
        self.p = p
        self.tour_n = tour_n
        self.func = func
        self.shape = shape
        self.x_best = [0]*shape
        self.y_best = 0
        self.x_history = []
        self.T_history = [self.T]
        self.m, self.n, self.quench = kwargs.get('m', 1), kwargs.get('n', 1), kwargs.get('quench', 1)
        self.lower, self.upper = kwargs.get('lower', -10), kwargs.get('upper', 10)
        self.c = self.m * np.exp(-self.n * self.quench)


    def xrange(self, xmin: np.ndarray, xmax: np.ndarray):  # 输入x范围，尚未考虑多维情形
        self.xmin = xmin
        print('x的下界是'+str(xmin))
        self.xmax = xmax
        print('x的上界是'+str(xmax))

    def init_pop(self):
        pop1 = []
        x_init = [10,10]
        for i in range(self.pop):
            pop1.append(self.fast_get(x_init))
        return pop1

    def judge(self, df):
        if df < 0:  # 新解 < 原解 ---> 直接接受
            return True
        else:
            pp = np.exp(-1 * (df / self.T))  # 新解 > 原解 ---> 计算接受概率p
            rand_p = random.random()  # 产生 rand_p ~ Uniform(0, 1)
            if pp > rand_p:
                return True
            else:
                return False

    def fast_get(self,x):
        r = np.random.uniform(-1, 1, size=self.shape)
        xc = np.sign(r) * self.T * ((1 + 1.0 / self.T) ** np.abs(r) - 1.0)
        x_new = x + xc * (self.upper - self.lower)
        return x_new


    def generate(self,old_pop):
        x_new = []

        for i in range(len(old_pop)):
            while len(x_new) < (i+1)*self.new_pop:
                xnew = self.fast_get(old_pop[i])

                # Metropolis准则
                df = self.func(xnew) - self.func(old_pop[i])  # 计算函数值差值
                if self.judge(df):
                    x_new.append(xnew)

        return x_new

    def tournament(self,x):
        """
        计算每个点的适应值，选择N个点作为生存集（GA）锦标赛。
        如果适应值最大的点被舍弃了，那么就随机舍弃一个点，又把它加回来。
        处理为先把适应值最大的点加进去
        """
        survive = []
        x_cur = x
        y_cur = [self.func(xx) for xx in x]
        best_index = y_cur.index(min(y_cur))
        self.x_best = x_cur[best_index]
        self.y_best = y_cur[best_index]
        survive.append(self.x_best)  # 先把最好的点放进去
        del(x_cur[best_index])  # 把最好的点删了，其他的进行锦标赛，最后选出self.pop-1个点

        #     进行选择（锦标赛方法）
        #     choose k (the tournament size) individuals from the population at random
        #     choose the best individual from the tournament with probability p
        #     choose the second best individual with probability p*(1-p)
        #     choose the third best individual with probability p*((1-p)^2)
        #     and so on

        fitness = [self.y_best - self.func(xx) for xx in x_cur]
        num_individuals = len(fitness)
        indices = list(range(len(fitness)))
        selected_indices = []
        selection_size = self.pop - 1

        while (len(selected_indices) < selection_size):
            np.random.shuffle(indices)
            idx_tournament = indices[0:self.tour_n]
            fit = []
            for i in range(len(idx_tournament)):
                fit.append(fitness[idx_tournament[i]])
            maxindex = fit.index(max(fit))
            winner = idx_tournament[maxindex]
            r = random.random()
            if r < self.p:
                selected_indices.append(winner)
            selected_indices = list(set(selected_indices))



        for i in range(len(selected_indices)):
            survive.append(x_cur[selected_indices[i]])

        return survive

    def cool(self):
        # fast:
        self.T = self.T_max * np.exp(-self.c * (self.cur_g * 10) ** self.quench)
        # self.T = self.T * 0.7
        self.T_history.append(self.T)

    def disp(self):
        if self.shape == 1:
            pass

        elif self.shape == 2:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x0_list = [[0 for col in range(len(self.x_history[0]))] for row in range(len(self.x_history))]
            x1_list = [[0 for col in range(len(self.x_history[0]))] for row in range(len(self.x_history))]
            for i in range(len(self.x_history)):
                for j in range(len(self.x_history[i])):
                    x0_list[i][j] = self.x_history[i][j][0]
                    x1_list[i][j] = self.x_history[i][j][1]
            x_min, x_max = min(min(row) for row in x0_list), max(max(row) for row in x0_list)
            y_min, y_max = min(min(row) for row in x1_list), max(max(row) for row in x1_list)
            x = np.arange(x_min, x_max, 0.01)
            y = np.arange(y_min, y_max, 0.01)
            X, Y = np.meshgrid(x, y)
            Z = self.func(np.array([X, Y]))
            ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.3)

            for i in range(len(self.x_history)):
                x0, x1, t = x0_list[i], x1_list[i], self.T_history[i]
                print(x0)
                y = [self.func(x) for x in self.x_history[i]]
                plt.title("当前温度 ："+str(t))
                p = ax.scatter(x0, x1, y, c='b')
                plt.pause(0.5)
                if i != len(self.x_history) - 1:
                    p.remove()

            plt.xlabel("x0")
            plt.ylabel("x1")
            plt.show()

    def main(self):
        pop1 = self.init_pop()
        self.x_history.append(pop1)
        while self.T > self.T_min:

            print('----------------当前迭代代数是' + str(self.cur_g) + '-------------------')
            new_pop = self.generate(pop1)
            pop1 = self.tournament(new_pop)
            self.x_history.append(pop1)
            self.cool()
            print('当前温度'+str(self.T))
            self.cur_g = self.cur_g + 1

            sur_pop_y = [self.func(xx) for xx in pop1]
            best_index = sur_pop_y.index(min(sur_pop_y))
            self.x_best = pop1[best_index]
            self.y_best = sur_pop_y[best_index]




if __name__ == "__main__":
    start = time.time()  # 开始计时
    x_init = np.array([10, 10])  # 设置初始点（行向量）
    x_min = np.array([-10, -10])
    x_max = np.array([10, 10])
    demo = GSA(func=Rastrigrin,T_max=1,T_min=1e-5,pop=30,new_pop=15,cur_g=1,p=0.9,tour_n=15,shape=2)
    demo.xrange(x_min, x_max)
    demo.main()
    demo.disp()
    end = time.time()  # 结束计时
    print("Duration time: %0.3f" % (end - start))  # 打印运行时间
    print('iterations:' + str(demo.cur_g))
    print('x_best:'+str(demo.x_best))
    print('y_best:'+str(demo.y_best))




