import numpy as np
import matplotlib.pyplot as plt
import random


#定义目标函数
def obj_fun(x):
    return x + 10*np.sin(5*x) + 7*np.cos(4*x)

class SimulatedAnnealing: #定义模拟退火类
    def __init__(self, f): #传入目标函数
        self.function = f
        self.tmp = 1e5
        self.tmp_min = 1e-4
        self.alpha = 0.98
        self.k = 1.15

    def setparameter(self, k):
        self.k = k

    def settmp(self, tmp):
        self.tmp = tmp

    def xrange(self, xmin, xmax): #输入x范围，尚未考虑多维情形
        self.xmin = xmin
        self.xmax = xmax

    def generatex(self, x_old):
        delta = (random.random() - 0.5)*(self.xmax-self.xmin) * self.k
        x_new = x_old + delta
        if x_new < (self.xmin + 10e-4) or (10e-4 + x_new) > self.xmax:
            x_new = x_new - 2 * delta
        return x_new

    def judge(self, dE, currtmp):
        if dE < 0:
            return True
        else:
            P = np.exp(-1*(dE/currtmp))
            randnum = random.random()
            if P > randnum:
                return True
            else:
                return False

    def main(self):
        x_init = self.xmin + random.random() * (self.xmax - self.xmin)

        #初始化参数
        x_old = x_init
        s_old = self.function(x_init)
        currtmp = self.tmp
        xlist = [[x_init, currtmp]]
        counter = 0
        Max = x_init
        while currtmp > self.tmp_min:
            x_new = self.generatex(x_old)
            s_new = self.function(x_new)
            dE = s_new - s_old
            if self.judge(dE, currtmp):
                Max = x_new
                s_old = s_new
                x_old = x_new
                xlist.append([x_old, currtmp])

            if dE > 0:
                currtmp *= self.alpha
                x_old = Max
            else:
                counter += 1

        # self.draw(xlist)
        return x_old

    def draw(self, xlist): # 可视化
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(self.xmin, self.xmax, 0.001)
        vf = np.vectorize(self.function)
        y = vf(x)
        plt.plot(x, y)
        # for i in xlist:
        #     num, t = i[0], i[1]
        #     plt.title("currtmp = {}".format(str(t)))
        #     p = ax.scatter(num ,self.function(num), marker='o')
        #     plt.pause(0.03)
        #     p.remove()
        ax.scatter(xlist[-1][0], self.function(xlist[-1][0]), marker='o')

        plt.show()



if __name__ == "__main__":
    demo = SimulatedAnnealing(obj_fun)
    demo.xrange(0,9)
    x = np.arange(2,7,1)
    y = []
    for i in range(x.shape[0]): # 不同温度下的收敛情况
        count = 0
        for j in range(1000):
            demo.settmp(10**x[i])
            rel = demo.main()
            if abs(rel - 0.89) < 0.1:
                count += 1
            print(count / 1000)
        y.append(count/1000)
    y = np.array(y)
    plt.plot(x,y)
    plt.ylim(0,1)
    plt.show()
    # count = 0
    # for i in range(1000):
    #     rel = demo.main()
    #     if abs(rel - 0.89) < 0.1:
    #         count += 1
    # print(count)
