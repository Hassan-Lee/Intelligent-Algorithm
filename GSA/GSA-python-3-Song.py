import numpy as np
import random
import time



# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 二维目标函数

# @pysnooper.snoop(prefix="Rastrigrin",output='F:/e/pycode/GSA.log')
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
    # @pysnooper.snoop(prefix="init",output='F:/e/pycode/GSA.log')
    def __init__(self,T_max=400,T_min=10,pop=10,new_pop=10,cur_g=1,p=0.9,tour_n=10,func=fun1,shape=1):
        """
        @param T_max: 最大温度
        @param T_min: 最小温度
        @param pop: 种群的个体数量
        @param new_pop: 每个个体生成的新解的数量
        @param cur_g: 当前迭代代数
        @param p: 锦标赛中的概率p
        @param tour_n: 一次锦标赛的选手数
        @param func: 函数
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
        self.shape = 1
        self.x_best = [1]*shape
        self.y_best = 1111

    # @pysnooper.snoop(prefix="xrange",output='F:/e/pycode/GSA.log')
    def xrange(self, xmin: np.ndarray, xmax: np.ndarray):  # 输入x范围，尚未考虑多维情形
        """
        @param xmin: x的下界
        @param xmax: x的上界
        """
        self.xmin = xmin
        # print('x的下界是'+str(xmin))
        self.xmax = xmax
        # print('x的上界是'+str(xmax))

    # @pysnooper.snoop(prefix="init_pop",output='F:/e/pycode/GSA.log')
    def init_pop(self):
        """
        生成初始种群
        @return: 初始种群
        """
        pop1 = []
        self.pop = int(self.pop)
        for i in range(self.pop):
            x_init = self.xmin + random.random() * (self.xmax - self.xmin)
            pop1.append(x_init)
        # print('初始的种群是'+str(pop1))
        return pop1

    # @pysnooper.snoop(prefix="judge",output='F:/e/pycode/GSA.log')
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

    # @pysnooper.snoop(prefix="generate",output='F:/e/pycode/GSA.log')
    def generate(self,old_pop):
        """
        根据上一次的种群，产生新解
        @param old_pop: 上一个种群
        """
        x_new = []
        for i in range(len(old_pop)):
            for j in range(self.new_pop):
                u = np.random.uniform(-1, 1, size=self.shape)  # 产生均匀分布的[Xi,……] u ~ Uniform(0, 1, size = d)
                xnew = old_pop[i] + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)

                # 接受准则，Metropolis准则
                df = self.func(xnew) - self.func(old_pop[i])  # 计算函数值差值
                if self.judge(df):  # 根据接受准则判断是否接受
                    x_new.append(xnew)

        return x_new

    def selection(self, x):
        # 计算每个个体的适应度值，并按照轮盘赌的方法挑选出N个，且适应度值最大的个体必须在其中

        survive = []
        x_new = [0]*self.pop
        x_cur = x
        y_cur = [self.func(xx) for xx in x]
        best_index = y_cur.index(min(y_cur))
        self.x_best = x_cur[best_index]
        self.y_best = y_cur[best_index]
        fitness = [self.y_best - self.func(xx) for xx in x_cur]

        fitness_sum = [0] * self.pop
        for i in range(self.pop):
            if i == 0:
                fitness_sum[i] = fitness_sum[i] + fitness[i]
            else:
                fitness_sum[i] = fitness_sum[i - 1] + fitness[i]

        for i in range(self.pop):
            r = np.random.random() * fitness_sum[self.pop - 1]
            first = 0
            last = self.pop
            mid = round((first + last) / 2)  # mid是种群数/2的那个位置
            idx = -1

            while first <= last and idx == -1:
                if r > fitness_sum[mid]:
                    first = mid
                elif r < fitness_sum[mid]:
                    last = mid
                else:
                    idx = mid
                    break
                mid = round((first + last) / 2)
                if last - first == 1:
                    idx = last
                    break
            x_new[i] = x_cur[idx]

        p = self.new_pop - 1
        for i in range(p):
            survive.append(x_new[i])
        survive.append(self.x_best)
        return survive

    # @pysnooper.snoop(prefix="tournament",output='F:/e/pycode/GSA.log')
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
            # print('这场锦标赛的参赛选手是'+str(idx_tournament))
            fit = []
            for i in range(len(idx_tournament)):
                fit.append(fitness[idx_tournament[i]])
            maxindex = fit.index(max(fit))
            winner = idx_tournament[maxindex]
            r = random.random()
            if r < self.p:
                selected_indices.append(winner)
                # print(str(winner)+'被加入')
            selected_indices = list(set(selected_indices))
            # print('去重后的是'+str(selected_indices))
            # print('总共有' + str(num_individuals) + '个个体')
            # print('现在selected的size是' + str(len(selected_indices)))
            # print('我们需要' + str(selection_size))


        for i in range(len(selected_indices)):
            survive.append(x_cur[selected_indices[i]])

        return survive

    # @pysnooper.snoop(prefix="cool",output='F:/e/pycode/GSA.log')
    def cool(self):
        self.T = self.T * 0.7
        # self.T = self.T_max * math.exp(0.7)

    # @pysnooper.snoop(prefix="main",output='F:/e/pycode/GSA.log')
    def main(self):
        pop1 = self.init_pop()
        while self.T > self.T_min:
            print('----------------当前迭代代数是' + str(self.cur_g) + '-------------------')
            new_pop = self.generate(pop1)
            pop1 = self.selection(new_pop)
            self.cool()
            print('当前温度'+str(self.T))
            self.cur_g = self.cur_g + 1

            sur_pop_y = [self.func(xx) for xx in pop1]
            best_index = sur_pop_y.index(min(sur_pop_y))
            self.x_best = pop1[best_index]
            self.y_best = sur_pop_y[best_index]


    # def disp(self):



def run(func=Rastrigrin,T_max=10000,T_min=10,pop=100,new_pop=100,cur_g=1,p=0.9,tour_n=5,shape=2):
    start = time.time()  # 开始计时
    x_init = np.array([10, 10])  # 设置初始点（行向量）
    x_min = np.array([-10, -10])
    x_max = np.array([10, 10])
    demo = GSA(T_max,T_min,pop,new_pop,cur_g,p,tour_n,func,shape)
    demo.xrange(x_min, x_max)
    demo.main()
    end = time.time()  # 结束计时
    print("Duration time: %0.3f" % (end - start))  # 打印运行时间
    print('iterations:' + str(demo.cur_g))
    print('x_best:'+str(demo.x_best))
    print('y_best:'+str(demo.y_best))

    return demo.x_best, demo.y_best


# start = time.time()  # 开始计时
# count = 0
# for i in range(10):
#     rel = run()[0]
#     if abs(rel[0] - (-0.24)) < 0.01:
#         count += 1
# print(count)
# end = time.time()  # 结束计时
# print("Duration time: %0.3f" % (end - start))  # 打印运行时间

# 手动网格搜索调参
T_max = [1,100,200,300,400,500,600,700,800,900,1000]
T_min = []
for i in range(len(T_max)):
    tmin = T_max[i]/((0.7)**10)
    T_min.append(tmin)
pop = list(range(0,500,10))
new_pop = list(range(0,500,10))
p = [0.5,0.6,0.7,0.8,0.9,0.95]
tour_n = list(range(2,80,3))
result = []
for i in range(len(T_max)):
    tmax = T_max[i]
    tmin = T_min[i]
    for j in pop:
        for k in new_pop:
            for l in p:
                for m in tour_n:
                    x_best,y_best = run(T_max=tmax, T_min=tmin, pop=int(j), new_pop=int(k), cur_g=1, p=l, tour_n=int(m), func=Rastrigrin, shape=2)
                    result.append([y_best,tmax,tmin,j,k,l,m])

output = open('/home/admin/Intelligent-Algorithm/GSA/data.xlsx','w')
output.write('y_best\tT_max\tT_min\tpop\tnew_pop\tp\ttour_n\n')
for i in range(len(result)):
    for j in range(len(result[i])):
        output.write(str(result[i][j]))  #write函数不能写int类型的参数，所以使用str()转化
        output.write('\t')  #相当于Tab一下，换一个单元格
    output.write('\n')    #写完一行立马换行
output.close()


