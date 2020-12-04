import numpy as np
import matplotlib.pyplot as plt
import copy as cp

def function(x):
    """

    :param x: 自变量
    :return: y: 函数值
    """
    return -(x+10*np.sin(5*x)+7*np.cos(4*x))

    #x*np.cos(5*np.pi*x)+3.5

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
        self.chromosome_size = chromosome_size
        self.left = left
        self.right = right
        self.generation = generation
        self.G = 1  # 当前代数
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.elitism = elitism

        self.population = np.zeros((self.population_size,self.chromosome_size),dtype=np.float)

        self.x0 = 0
        self.best_fitness = 0  # 最高适应度个体的适应度
        self.best_generation = 0  # 最高适应度个体所在的代数
        self.best_individual = np.zeros(chromosome_size,dtype=np.float)  # 最高适应度个体的染色体
        self.fitness_value = np.zeros(self.population_size,dtype=np.float)  # 当前代每个个体的适应度值
        self.fitness_value_max = np.zeros(self.generation, dtype=np.float) # 每代适应度最大值
        self.fitness_average = np.zeros(self.generation, dtype=np.float)  # 每代平均适应度值

    def init(self):
        """
        初始化种群,随机生成一系列染色体.
        :return:
        """
        for i in range(self.population_size):
            for j in range(self.chromosome_size):
                self.population[i][j] = round(np.random.random())

    def fitness(self):
        """
        计算各个个体的适应度
        :return:
        """
        self.fitness_value = np.zeros(self.population_size,dtype=np.float)

        # 对染色体进行解码
        for i in range(self.population_size):
            for j in range(self.chromosome_size):
                if self.population[i][j] == 1:
                    self.fitness_value[i] += 2**j

            self.fitness_value[i] = self.left + self.fitness_value[i]*(self.right-self.left)/(2**self.chromosome_size-1)
            self.fitness_value[i] = function(self.fitness_value[i])

    def rank(self):
        '''
        根据适应度对种群进行排名
        :return:
        '''
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
        for i in range(0,self.population_size,2):
            if np.random.random() < self.cross_rate:
                cross_position = round(np.random.random()*self.chromosome_size)
                if cross_position == self.chromosome_size:
                    continue
                for j in range(cross_position,self.chromosome_size):
                    temp = cp.deepcopy(self.population[i+1][j])
                    self.population[i+1][j] = self.population[i][j]
                    self.population[i][j] = temp

    def mutation(self):
        """
        变异
        :return:
        """
        for i in range(self.population_size):
            if np.random.random() < self.mutate_rate:
                mutate_position = round(np.random.random()*self.chromosome_size)
                if mutate_position == self.chromosome_size:
                    continue
                self.population[i][mutate_position] = 1 - self.population[i][mutate_position]

    def graph_draw(self):
        q = 0
        for i in range(self.chromosome_size):
            if self.best_individual[i] == 1:
                q += 2 ** i
        self.x0 = self.left + q * (self.right - self.left) / (2 ** self.chromosome_size - 1)
        fig, axes = plt.subplots(1,2,figsize=(10,3),tight_layout=True)
        x0 = np.arange(self.left,self.right,0.01)
        y0 = function(x0)
        axes[0].plot(x0,y0)
        axes[0].scatter(self.x0,self.best_fitness,marker='o')
        x1 = np.arange(1,self.generation+1,1)
        axes[1].plot(x1,self.fitness_average)
        axes[1].scatter(x1,self.fitness_value_max)

        plt.show()

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


test = GA(0,20,40,15,30,0.6,0.1,True)
print(test.genetic_algorithm())
