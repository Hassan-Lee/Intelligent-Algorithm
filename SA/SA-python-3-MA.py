import numpy as np
import matplotlib.pyplot as plt
import random
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 定义目标函数
def obj_fun(x: np.ndarray):
    return abs(0.2 * x[0]) + 10 * np.sin(5 * x[0]) + 7 * np.cos(4 * x[0])


def obj_fun2(x: np.ndarray):
    return (1 - x[0]) ** 2 + 100 * (x[0] - x[1] ** 2) ** 2

def Easom(x: np.ndarray):
    return -1*np.cos(x[0])*np.cos(x[1])*np.exp(-1*(x[0]-np.pi)**2-(x[0]-np.pi)**2)

def obj_fun_test1(x:np.ndarray):
    return -1 * (10 + np.sin(1/x)/((x - 0.16)**2+0.1))

def Bohachevsky(x:np.ndarray):
    return x[0]**2 + x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) + 0.3*np.cos(4*np.pi*x[1]) + 0.3

def Rastrigrin(x:np.ndarray):
    return 20+x[0]**2-10*np.cos(2*np.pi*x[0])+x[1]**2-10*np.cos(2*np.pi*x[1])

def Schaffersf6(x:np.ndarray):
    return 0.5+(np.sin(x[0]**2 + x[1]**2)**2 -0.5)/(1+0.001*(x[0]**2 + x[1]**2)**2)**2

def Shubert(x:np.ndarray):
    s1 = 0
    s2 = 0
    for i in range(1,6):
        s1 += i*np.cos((i+1)*x[0] + i)
        s2 += i*np.cos((i+1)*x[1] + i)
    return s1+s2


class SimulatedAnnealing:  # 定义模拟退火类
    def __init__(self, f):  # 传入目标函数
        self.function = f
        self.tmp = 1
        self.tmp_min = 1e-5
        self.alpha = 0.99
        self.k = 1

    def setalpha(self, alpha):
        self.alpha = alpha

    def setparameter(self, k):
        self.k = k

    def settmp(self, tmp):
        self.tmp = tmp

    def xrange(self, xmin: np.ndarray, xmax: np.ndarray):  # 输入x范围，尚未考虑多维情形
        self.xmin = xmin
        self.xmax = xmax

    def generatex(self, x_old, currtmp):
        if currtmp > 1e-2:
            self.k = 1
        elif 1e-3 < currtmp < 1e-2:
            self.k = 0.1
        elif 1e-4 < currtmp < 1e-3:
            self.k = 0.01
        else:
            self.k = 0.001
        newlist = []
        for i in range(1):
            x_new = np.zeros(shape=x_old.shape)
            for i in range(x_old.shape[0]):
                delta = (random.random() - 0.5) * (self.xmax[i] - self.xmin[i]) * self.k
                x_new[i] = x_old[i] + delta
                if x_new[i] < (self.xmin[i] + 10e-4) or (10e-4 + x_new[i]) > self.xmax[i]:
                    x_new[i] = x_new[i] - 2 * delta
            newlist.append([x_new, self.function(x_new)])
        newlist.sort(key=lambda x:x[1])
        return newlist[0][0]

    def judge(self, dE, currtmp):
        if dE < 0:
            return True
        else:
            P = np.exp(-1 * (dE / currtmp))
            randnum = random.random()
            if P > randnum:
                return True
            else:
                return False

    def main(self):
        x_init = self.xmin + random.random() * (self.xmax - self.xmin)

        # 初始化参数
        x_old = x_init
        s_old = self.function(x_init)
        currtmp = self.tmp
        xlist = [[x_init, currtmp, 0]]
        counter = 0
        Max = x_init

        while currtmp > self.tmp_min:
            x_new = self.generatex(x_old, currtmp)
            s_new = self.function(x_new)
            dE = s_new - s_old
            if self.judge(dE, currtmp):
                s_old = s_new
                x_old = x_new
                xlist.append([x_old, currtmp, counter])

            if dE > 0:

                currtmp *= self.alpha
            else:
                counter += 1
        # print(counter)
        self.draw(xlist)
        # self.plot(xlist)
        return x_old, self.function(x_old)

    def draw(self, xlist):  # 可视化
        if xlist[0][0].shape[0] == 1:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            x = np.arange(self.xmin[0], self.xmax[0], 0.001)
            y = np.zeros(shape=x.shape)
            for index in range(x.shape[0]):
                y[index] = self.function(np.array([x[index]]))
            plt.plot(x, y)
            for i in range(len(xlist)):
                num, t = xlist[i][0], xlist[i][1]
                plt.title("currtmp = {}".format(str(t)))
                p = ax.scatter(num, self.function(num), marker='o')
                plt.pause(0.01)
                if i != len(xlist) - 1:
                    p.remove()

            plt.show()
        elif xlist[0][0].shape[0] == 2:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x = np.arange(self.xmin[0], self.xmax[0], 0.01)
            y = np.arange(self.xmin[1], self.xmax[1], 0.01)
            X, Y = np.meshgrid(x, y)
            Z = self.function(np.array([X, Y]))
            ax.plot_surface(X, Y, Z, cmap='GnBu', alpha=0.75)
            # ax.scatter(1, -1, self.function(np.array([1, -1])), c='r', marker='*')
            # ax.scatter(1, 1, self.function(np.array([1, 1])), c='r', marker='*')

            # 绘制等高线
            ax.contour(X, Y, Z, zdir='x', offset=-50)
            for i in range(len(xlist)):
                num, t = xlist[i][0], xlist[i][1]
                plt.title("currtmp = {}".format(str(t)))
                p = ax.scatter(num[0], num[1], self.function(np.array([num[0], num[1]])), c='b', marker='o')
                plt.pause(0.01)
                if i != len(xlist) - 1:
                    p.remove()

            plt.show()

    def plot(self, xlist):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        y = []
        counters = []
        tmps = []
        for i in xlist:
            y.append(self.function(np.array([i[0][0],i[0][1]])))
            counters.append(i[2])
            tmps.append(i[1])
        y = np.array(y)
        counters = np.array(counters)
        plt.xlabel("迭代次数")
        plt.ylabel("curry and currtmp")
        plt.vlines(max(counters) / 2, 0, max(y), colors="c", linestyles="dashed")
        plt.plot(counters, y)
        plt.plot(counters, tmps)
        plt.show()


if __name__ == "__main__":

    #普通运行一次（一般是看结果或者调可视化）
    demo = SimulatedAnnealing(Bohachevsky)
    xmin = np.array([-20,-20])
    xmax = np.array([20,20])
    demo.xrange(xmin,xmax)

    print(demo.main())

    # 调参
    # demo = SimulatedAnnealing(obj_fun)
    # xmin = np.array([-20])
    # xmax = np.array([20])
    # demo.xrange(xmin,xmax)
    #
    # x = np.arange(0.9,1,0.01)
    # y = []
    # for i in range(x.shape[0]):
    #     count = 0
    #     for j in range(1000):
    #         demo.setalpha(x[i])
    #         rel = demo.main()
    #         if abs(rel - (-5.39107190851645)) < 0.01:
    #             count += 1
    #     print(count / 1000)
    #     y.append(count / 1000)
    # y = np.array(y)
    # plt.plot(x,y)
    # plt.ylim(0,1)
    # plt.show()
    #
    # 时间计算
    # start = time.time()
    # demo = SimulatedAnnealing(obj_fun_test1)
    # xmin = np.array([-0.5])
    # xmax = np.array([0.5])
    # demo.xrange(xmin,xmax)
    # count = 0
    # for i in range(1000):
    #     rel = demo.main()
    #     if abs(rel - 0.12748411) < 0.01:
    #         count += 1
    # print(count)
    # end = time.time()
    # print("Duration time: %0.3f" % (end - start))
    #
    # 普通运行一次（一般是看结果或者调可视化）
    # demo = SimulatedAnnealing(obj_fun2)
    # xmin = np.array([-2, -2])
    # xmax = np.array([2, 2])
    # demo.xrange(xmin, xmax)
    # print(demo.main())

    # x = np.arange(-2,2, 0.01)
    # y = x
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # X,Y = np.meshgrid(x,y)
    # Z = (1-X)**2 + 100*(X-Y**2)**2
    # ax.plot_surface(X, Y, Z, cmap='rainbow')
    # plt.show()


    # demo = SimulatedAnnealing(Easom)
    # xmin = np.array([-10, -10])
    # xmax = np.array([10, 10])
    # demo.xrange(xmin, xmax)
    # print(demo.main())

    # demo = SimulatedAnnealing(obj_fun_test1)
    # xmin = np.array([-0.5])
    # xmax = np.array([0.5])
    # demo.xrange(xmin,xmax)
    #
    # print(demo.main())
