import numpy as np
import matplotlib.pyplot as plt
import copy as cp

def function(x):
    """

    :param x: 自变量
    :return: y: 函数值
    """

    return abs(0.2 * x) + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)

"""
def functiontt(x):
    

    :param x: 自变量,为二维向量
    :return: 函数值
    
    return 20 + x[0] ** 2 + x[1] ** 2 - 10 * np.cos(2*np.pi*x[0]) - 10 * np.cos(2*np.pi*x[1])

def functiont(x):
    return 20 + x[0]**2 + x[1]**2 - 10*np.cos(2*np.pi*x[0]) - 10 * np.cos(2*np.pi*x[1])
    #1/4000 * (x[0]**2 + x[1]**2) - np.cos(x[0])*np.cos(x[1]/np.sqrt(2)) + 1
"""
class GA:
    def __init__(self,left,right,population_size,chromosome_size,generation,cross_rate,mutate_rate,select_mode='roulette',elitism = True):
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
        self.chromosome_size = chromosome_size
        self.left = left #暂时的
        self.right = right
        self.generation = generation
        self.G = 1  # 当前代数
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.elitism = elitism

        self.population = np.zeros([self.population_size,chromosome_size],dtype=np.float)
        self.mode = chromosome_size
        self.x0 = np.zeros_like(self.population[0],dtype=np.float)
        self.best_fitness = 0  # 最高适应度个体的适应度
        self.best_generation = 0  # 最高适应度个体所在的代数
        self.best_individual = np.zeros_like(self.population[0])  # 最高适应度个体的染色体
        self.fitness_value = np.zeros(self.population_size, dtype=np.float)  # 当前代每个个体的适应度值
        self.fitness_value_interval = np.zeros([self.generation, 2], dtype=np.float) # 每代初始适应度范围
        self.fitness_value_max = np.zeros(self.generation, dtype=np.float)  # 每代适应度最大值
        self.fitness_average = np.zeros(self.generation, dtype=np.float)  # 每代平均适应度值

        self.select_mode = select_mode


    def init(self):
        """
        初始化种群,随机生成一系列染色体.
        :return:
        """
        for i in range(self.population_size):
            for j in range(self.chromosome_size):
                self.population[i][j] = np.random.random() * (self.right[j] - self.left[j])

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
        return (x0 - left)/(right - left)

    def fitness(self):
        """
        计算各个个体的适应度,先转换成二进制再计算,转换时不改变原编码.
        :return:
        """

        for i in range(self.population_size):
            self.fitness_value[i] = function(self.population[i])
        if self.G == 1:
            self.best_fitness = self.fitness_value[round(np.random.random()*(self.population_size-1))]




    def rank(self):
        """
        根据适应度对种群进行排名
        :return:
        """
        if self.select_mode == 'roulette':
            self.fitness_value_interval[self.G] = cp.deepcopy([min(self.fitness_value),max(self.fitness_value)])
            self.fitness_value = self.fitness_trans(self.fitness_value)
        else: pass
        self.fitness_sum = np.zeros(self.population_size, dtype=np.float)
        # bubblesort
        for i in range(self.population_size):
            min_index = i
            for j in range(i,self.population_size):
                #print(j,min_index)
                if self.fitness_value[j] < self.fitness_value[min_index]:
                    min_index = j
            if min_index != i:
                temp = cp.deepcopy(self.fitness_value[min_index])
                self.fitness_value[min_index] = self.fitness_value[i]
                self.fitness_value[i] = temp
                temp = cp.deepcopy(self.population[min_index])
                self.population[min_index] = self.population[i]
                self.population[i] = temp

        # 计算物种适应值之和,之后轮盘赌.否则线性排名.
        if self.select_mode == 'roulette':
            for i in range(self.population_size):
                if i == 0:
                    self.fitness_sum[i] = self.fitness_sum[i] + self.fitness_value[i]
                else:
                    self.fitness_sum[i] = self.fitness_sum[i-1] + self.fitness_value[i]
            self.fitness_average[self.G] = self.fitness_sum[self.population_size-1]/self.population_size
            self.fitness_value_max[self.G] = self.fitness_value[-1]

        if self.fitness_value[0] <= self.best_fitness:
            self.best_fitness = self.fitness_value[0]
            self.best_generation = self.G
            self.best_individual = cp.deepcopy(self.population[0])


    def selection(self):
        """
        根据适应度进行轮盘赌选择
        :return:
        """
        population_new = np.zeros((self.population_size,self.chromosome_size),dtype=np.float)
        # 轮盘赌
        if self.select_mode == 'roulette':
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
        # 线性排名,几何分布
        else:
            for i in range(self.population_size):
                num = np.random.random()
                for j in range(self.population_size):
                    if (1-num) >= (2/3)**(j+1):
                        # 条件成立就是选第j+1名
                        population_new[i] = self.population[j]
                        break
            if self.elitism:
                p = self.population_size - 1
                for i in range(1,p+1):
                    self.population[i] = population_new[i]

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
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (1 + 1))
                elif u <= 1:
                    beta = (1 / (2 - 2*u)) ** (1 / (1 + 1))
                else: beta = 0
                c1 = 1 / 2 * (self.population[i][j] + self.population[i+1][j]) - 1 / 2 * beta * (self.population[i+1][j] - self.population[i][j])
                c2 = 1 / 2 * (self.population[i][j] + self.population[i][j]) + 1 / 2 * beta * (self.population[i+1][j] - self.population[i][j])
                self.population[i][j] = c1
                self.population[i+1][j] = c2


    def mutation(self):
        """
        变异
        :return:
        """
        for i in range(self.population_size):
            if np.random.random() < self.cross_rate:
                mutation_position = round(np.random.random() * self.chromosome_size)
                if mutation_position >= self.chromosome_size:
                    pass
                elif self.chromosome_size == 1:
                    if round(np.random.random()) == 0:
                        self.population[i] = self.population[i] + (
                                    self.right[mutation_position] - self.population[i]) * (
                                                                            1 - np.random.random() ** (
                                                                                (1 - (self.G / self.generation)) ** 2))
                    else:
                        self.population[i] = self.population[i] - (
                                    self.population[i] - self.left[mutation_position]) * (1 - np.random.random() ** (
                                    (1 - (self.G / self.generation)) ** 2))
                else:
                    if round(np.random.random()) == 0:
                        self.population[i][mutation_position] = self.population[i][mutation_position] + (self.right[mutation_position] - self.population[i][mutation_position]) * (1 - np.random.random() ** ((1-(self.G/self.generation))**2) )
                    else:
                        self.population[i][mutation_position] = self.population[i][mutation_position] - (self.population[i][mutation_position] - self.left[mutation_position]) * (1 - np.random.random() ** ((1-(self.G/self.generation))**2) )


    def graph_draw(self):
        self.x0 = self.best_individual
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
            x = np.arange(self.left[0], self.right[0], 0.3)
            y = np.arange(self.left[1], self.right[1], 0.3)
            X, Y = np.meshgrid(x, y)
            Z = function(np.array([X,Y]))
            plt.xlabel('x')
            plt.ylabel('y')
            ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow',alpha=0.9)
            ax1.scatter(self.x0[0], self.x0[1], function(np.array(self.x0)), marker='*', color='black')
            #ax1.scatter(0,0,120,marker='o',color='yellow')
            ax1.view_init(60,30)
            ax2 = fig.add_subplot('122')
            x1 = np.arange(1,self.generation+1,1)
            ax2.set_ylim(0,1)
            ax2.plot(x1, self.fitness_average)
            ax2.scatter(x1, self.fitness_value_max)
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


test = GA(np.array([-5]), np.array([5]), 80, 1, 80, 0.6, 0.1,'linrank', elitism=True)
print(test.run())
