"""
    改变为格雷码的编码方式.
    仍使用最小值.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy as cp

def function(x):
    """

    :param x: 自变量
    :return: y: 函数值
    """

    return 10 + np.sin(1/x)/((x - 0.16)**2+0.1)
    # -(x+10*np.sin(5*x)+7*np.cos(4*x))

    # x*np.cos(5*np.pi*x)+3.5


def functiont(x):
    """

    :param x: 自变量,为二维向量
    :return: 函数值
    """
    return 20 + x[0] ** 2 + x[1] ** 2 - 10 * np.cos(2*np.pi*x[0]) - 10 * np.cos(2*np.pi*x[1])


class GA:
    def __init__(self,left,right,population_size,chromosome_size,generation,cross_rate,mutate_rate,elitism = True):
        """

        :param left: 左端点
        :param right: 右端点
        :param population_size: 种群数量
        :param chromosome_size: 染色体数量
        :param generation: 代数
        :param cross_rate: 交叉概率
        :param mutate_rate: 变异概率
        :param elitism: 是否留取上一代适应度最大的个体
        """
        self.population_size = population_size
        a = chromosome_size[0] * chromosome_size[1]
        self.chromosome_size = a
        self.left = left #暂时的
        self.right = right
        self.generation = generation
        self.G = 1  # 当前代数
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.elitism = elitism

        self.population = np.zeros((self.population_size,self.chromosome_size),dtype=np.float)
        self.mode = chromosome_size[1]
        self.x0 = np.zeros([chromosome_size[1],1],dtype=np.float)
        self.best_fitness = 0  # 最高适应度个体的适应度
        self.best_generation = 0  # 最高适应度个体所在的代数
        self.best_individual = np.zeros(self.chromosome_size, dtype=np.float)  # 最高适应度个体的染色体
        self.fitness_value = np.zeros(self.population_size, dtype=np.float)  # 当前代每个个体的适应度值
        self.fitness_value_interval = np.zeros([self.generation, 2], dtype=np.float) # 每代初始适应度范围
        self.fitness_value_max = np.zeros(self.generation, dtype=np.float)  # 每代适应度最大值
        self.fitness_average = np.zeros(self.generation, dtype=np.float)  # 每代平均适应度值


    def init(self):
        """
        初始化种群,随机生成一系列染色体.
        :return:
        """
        for i in range(self.population_size):
            for j in range(self.chromosome_size):
                self.population[i][j] = round(np.random.random())

    def reinit(self):
        """
        再次初始化,清除所有属性.
        :return:
        """
        self.population = np.zeros((self.population_size,self.chromosome_size),dtype=np.float)
        self.x0 = 0
        self.best_fitness = 0  # 最高适应度个体的适应度
        self.best_generation = 0  # 最高适应度个体所在的代数
        self.best_individual = np.zeros(self.chromosome_size,dtype=np.float)  # 最高适应度个体的染色体
        self.fitness_value = np.zeros(self.population_size,dtype=np.float)  # 当前代每个个体的适应度值
        self.fitness_value_max = np.zeros(self.generation, dtype=np.float) # 每代适应度最大值
        self.fitness_average = np.zeros(self.generation, dtype=np.float)  # 每代平均适应度值

    def fitness_trans(self,x0):
        """
        对原函数进行复合嵌套,构造适应度函数
        :param x0:
        :return:
        """
        left = self.fitness_value_interval[self.G][0]
        right = self.fitness_value_interval[self.G][1]
        return (x0 - left + 0.1)/(right - left + 0.1)

    def graytobin(self,num:np.ndarray):
        shape = np.shape(num)
        length = shape[0]
        newnum = np.zeros_like(num)
        if len(shape) == 1:
            newnum[-1] = num[-1]
            for i in range(length - 2, -1, -1):
                newnum[i] = int(num[i + 1]) ^ int(num[i])
        else:
            for i in range(shape[0]):
                newnum[i][-1] = num[i][-1]
                for j in range(shape[1]-2,-1,-1):
                    newnum[i][j] = int(num[i][j+1]) ^ int(num[i][j])
        return newnum

    def fitness(self):
        """
        计算各个个体的适应度,先转换成二进制再计算,转换时不改变原编码.
        :return:
        """
        fitness_value = np.zeros((self.population_size, self.mode), dtype=np.float)
        bin_code = self.graytobin(self.population)
        # 格雷码转换成二进制码,生成新的数组来计算

        for i in range(self.population_size):
            for j in range(self.mode):
                for k in range(j * self.chromosome_size // self.mode, (j + 1) * self.chromosome_size // self.mode):
                    if bin_code[i][k] == 1:
                        fitness_value[i][j] += 2 ** (k - j * self.chromosome_size // self.mode)
            for j in range(self.mode):
                fitness_value[i][j] = self.left[j] + fitness_value[i][j] * (self.right[j] - self.left[j]) / (
                            2 ** (self.chromosome_size // self.mode) - 1)
            self.fitness_value[i] = function(fitness_value[i])
        #print(self.fitness_value)



    def rank(self):
        """
        根据适应度对种群进行排名
        :return:
        """

        self.fitness_value_interval[self.G] = cp.deepcopy([max(self.fitness_value),min(self.fitness_value)])
        self.fitness_value = self.fitness_trans(self.fitness_value)
        self.fitness_sum = np.zeros(self.population_size,dtype=np.float)
        # bubblesort
        for i in range(self.population_size):
            min_index = i
            for j in range(i,self.population_size):
                if self.fitness_value[j] < self.fitness_value[min_index]:
                    min_index = j
            if min_index != i:
                temp = cp.deepcopy(self.fitness_value[min_index])
                self.fitness_value[min_index] = self.fitness_value[i]
                self.fitness_value[i] = temp
                temp = cp.deepcopy(self.population[min_index])
                self.population[min_index] = self.population[i]
                self.population[i] = temp
        print(self.fitness_value)
        #self.fitness_value_interval[self.G] = cp.deepcopy([self.fitness_value[0],self.fitness_value[-1]])
        #self.fitness_value = self.fitness_trans(self.fitness_value)
        # 计算物种适应值之和,之后轮盘赌.
        for i in range(self.population_size):
            if i == 0:
                self.fitness_sum[i] = self.fitness_sum[i] + self.fitness_value[i]
            else:
                self.fitness_sum[i] = self.fitness_sum[i-1] + self.fitness_value[i]
        self.fitness_average[self.G] = self.fitness_sum[self.population_size-1]/self.population_size
        self.fitness_value_max[self.G] = self.fitness_value[-1]

        if self.fitness_value[self.population_size-1] >= self.best_fitness:
            self.best_fitness = self.fitness_value[self.population_size-1]
            self.best_generation = self.G
            self.best_individual = cp.deepcopy(self.population[self.population_size-1])


    def selection(self):
        """
        根据适应度进行轮盘赌选择
        :return:
        """
        population_new = np.zeros((self.population_size,self.chromosome_size),dtype=np.float)
        for i in range(self.population_size):
            r = np.random.random() * self.fitness_sum[self.population_size-1]
            first = 0
            last = self.population_size - 1
            mid = round((first+last)/2)
            idx = -1

            while first <= last and idx == -1:
                if r > self.fitness_sum[mid]:
                    first = mid
                elif r < self.fitness_sum[mid]:
                    last = mid
                else:
                    idx = mid
                    break
                mid = round((first+last)/2)
                if last - first == 1:
                    idx = last
                    break
            population_new[i] = self.population[idx]
        if self.elitism:
            p = self.population_size - 1
        else:
            p = self.population_size
        for i in range(p):
            self.population[i] = population_new[i]


    def crossover(self):
        """
        交叉
        :return:
        """
        for i in range(0, self.population_size, 2):
            for j in range(self.mode):
                cross_position = int(round(np.random.random() * (self.chromosome_size // self.mode)))
                # print(cross_position,np.random.random() * (self.chromosome_size // self.mode))
                if cross_position < self.chromosome_size // self.mode or cross_position > self.chromosome_size // self.mode:
                    # print(cross_position,(j + 1) * self.chromosome_size // self.mode)
                    for k in range(cross_position, (j + 1) * self.chromosome_size // self.mode):
                        temp = cp.deepcopy(self.population[i + 1][k])
                        self.population[i + 1][k] = self.population[i][k]
                        self.population[i][k] = temp


    def mutation(self):
        """
        变异
        :return:
        """
        for i in range(self.population_size):
            if np.random.random() < self.mutate_rate:
                mutate_position = int(round(np.random.random()*self.chromosome_size))
                if mutate_position == self.chromosome_size:
                    continue
                #print(mutate_position)
                self.population[i][mutate_position] = 1 - self.population[i][mutate_position]

    def graph_draw(self):
        q = np.zeros([self.mode,1])
        # 再次生成一个新的变量,存储转换而成的二进制码.
        best_individual = self.graytobin(self.best_individual)

        for i in range(self.mode):
            for j in range(i*self.chromosome_size//self.mode,(i+1)*self.chromosome_size//self.mode):
                if best_individual[j] == 1:
                    q[i] += 2 ** (j - i * self.chromosome_size//self.mode)
            self.x0[i] = self.left[i] + q[i] * (self.right[i] - self.left[i]) / (2 ** (self.chromosome_size // self.mode) - 1)
        if self.mode == 1:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3), tight_layout=True)
            x0 = np.arange(self.left[0], self.right[0], 0.01)
            y0 = function(x0)
            axes[0].plot(x0, y0)
            axes[0].scatter(self.x0, function(np.array(self.x0)), marker='o')
            x1 = np.arange(1, self.generation + 1, 1)
            axes[1].plot(x1, self.fitness_average)
            axes[1].scatter(x1, self.fitness_value_max)
            fig.show()
        elif self.mode == 2:
            fig = plt.figure(figsize=(10,3))
            ax1 = fig.add_subplot('121', projection='3d')
            x = np.arange(self.left[0], self.right[0], 0.25)
            y = np.arange(self.left[1], self.right[1], 0.25)
            X, Y = np.meshgrid(x, y)
            Z = function(np.array([X,Y]))
            plt.xlabel('x')
            plt.ylabel('y')
            ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow',alpha=0.1)
            ax1.scatter(self.x0[0], self.x0[1], function(np.array(self.x0)), marker='*', color='black')
            ax1.scatter(0,0,120,marker='o',color='yellow')
            ax1.view_init(60,30)
            plt.show()
        else:
            pass

    def genetic_algorithm(self):

        # 初始化种群
        self.init()

        # 进化过程
        for i in range(self.generation):
            self.G = i
            self.fitness()
            self.rank()

            self.selection()
            self.crossover()
            self.mutation()
        # 绘图
        self.graph_draw()
        return self.x0,self.best_individual,self.best_fitness,self.best_generation

    def run(self):
        x0, best_individual, best_fitness, best_generation = self.genetic_algorithm()
        self.reinit()
        return x0, best_individual, function(x0), best_generation


test = GA(np.array([-0.5]), np.array([0.5]), 80, np.array([16, 1]), 80, 0.6, 0.1, elitism=True)
print(test.run())
"""
numsum = 0
for i in range(1000):
    if test.run()[2] > 19.7:
        numsum += 1
    if numsum % 10 == 0:
        print(numsum)
        print('sign'+str(numsum//10))
numsum /= 1000
print(numsum)
"""