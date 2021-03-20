import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D


class Population(object):
    # 初始化种群，#pop_num表示该种群的数量，chromosome_length表示染色体的个数
    # 先将种群进行编码，X,Y均为18位，所以可以定义36位，前18位表示x,后18位表示y
    def __init__(self, pop_num, chromosome_length, left, right):
        self.pop_num = pop_num
        self.chromosome_length = chromosome_length
        bin_population = np.empty((pop_num, chromosome_length), dtype=np.uint)  # 创建一个二维空数组，表示种群
        for i in range(pop_num):
            for j in range(chromosome_length):
                bin_population[i][j] = random.randint(0, 1)  # 创建一个二进制编码的种群数组
        self.bin_population = bin_population
        self.left = left
        self.right = right
        self.bin_population = bin_population

    ##对种群二进制进行解码为十进制，其中left表示取值区间的最小值，right表示取值区间的最大值
    # deci_total表示种群中个体的十进制表示方式，true_deci表示该十进制在取值区间的具体大小
    def decode(self):
        a = self.chromosome_length  # a代表染色体基因长度
        b = int(a / 2)  # b代表X/Y的基因长度
        c = self.pop_num  # c代表种群个数
        pop1 = self.bin_population[:, 0:b]  # 取x的二进制
        pop2 = self.bin_population[:, b:a]  # 取y的二进制
        deci_population = []  # 创建一个一维数组，用来存放十进制的种群
        for i in range(len(pop1[:])):
            x_deci_total = 0
            for j in range(b):
                x_deci_total += pop1[i][j] * math.pow(2, j)  # 二进制转换成十进制
            deci_population.append(x_deci_total)
        for i in range(len(pop2[:])):
            y_deci_total = 0
            for j in range(b):
                y_deci_total += pop2[i][j] * math.pow(2, j)
            deci_population.append(y_deci_total)  # 将Y值的十进制数跟在X值后面
        deci_population = np.asarray(deci_population)
        true_pop = np.empty(len(deci_population))
        for i in range(len(true_pop)):
            true_deci = self.left + deci_population[i] * (self.right - self.left)/ (math.pow(2, b) - 1)  # 十进制数转换成对应区间的值
            true_pop[i] = true_deci
        return true_pop  # true_pop是解码后X,Y的真实值

    # 首先计算解码后每个个体的函数值,func_value代表函数值
    def function(self):
        true_pop = self.decode()  # 调用函数进行解码
        a1 = len(true_pop)
        b1 = int(len(true_pop) / 2)  # 种群个数
        pop1 = true_pop[0:b1]
        pop2 = true_pop[b1:a1]
        fitness = []
        for i in range(b1):
            func = math.sin(pop1[i]) * math.sin(pop2[i]) / pop1[i] / pop2[i]  # 计算函数值
            fitness.append(func)  # 将函数值储存在fitness函数组中
        return [fitness, pop1, pop2]

    # 1、轮盘选择法,其中func_p代表每一个个体占总和的概率，随机进行拨盘，被选中的个体进入下一代的种群中
    def selection(self):
       fitness = self.function()[0]  # 调用函数得到对应的函数值
       fitness = np.asarray(fitness)
       total = np.sum(fitness)  # 计算适应度的和
       if total == 0:
           total = 1  # 防止种群内个体一样使得total为0
       func_p = fitness / total  # 每一个个体被选择的概率
       func_accum = np.cumsum(func_p)  # 计算个体的累积概率
       selec = []
       if len(func_p[:]) > 1:
           for i in range(len(func_accum)):
               randomnum = np.random.random()  # 生成随机数，选择
               for j in range(len(func_accum)):
                   if (randomnum < func_accum[j]):
                       selec.append(copy.copy(self.bin_population[j, :]))
                       break
       self.bin_population = np.asarray(selec)

    # 2、锦标赛选择法,每次随机选择两个个体，在这其中选择适应度最好的一个进入下一代
    # def selection(self):
    #     fitness = self.function()[0]  # 调用函数得到对应的函数值
    #     fitness = np.asarray(fitness)
    #     selec = []
    #     for i in range(len(fitness[:])):
    #         k1 = random.randint(0, len(fitness[:])-1)
    #         k2 = random.randint(0, len(fitness[:])-1)            #随机选取两个，其中适应度高的进入下一代群体
    #         if fitness[k2] > fitness[k1]:
    #             selec.append(copy.copy(self.bin_population[k2, :]))
    #         else:
    #            selec.append(copy.copy(self.bin_population[k1, :]))
    #     self.bin_population = np.asarray(selec)

    # 种群交叉,其中pc是概率阈值，由他决定是否进行交叉，同时这里选择两个点进行交叉
    def crossover(self, pc):
        f = len(self.bin_population[:])
        new_pop = np.empty_like(self.bin_population)
        c = self.chromosome_length
        d = int(c / 2)
        for i in range(0, f, 2):
            if random.random() > pc:
                point1 = random.randint(0, d - 1)  # 选择第一个交叉点，在X基因中任选一个
                point2 = random.randint(d, c - 1)  # 选择第二个交叉点，在y基因中任选一个
                # 将point1后的位数进行交叉，point2后的位数进行交叉
                new_pop[i, 0:point1] = copy.copy(self.bin_population[i, 0:point1])
                new_pop[i, point1:d] = copy.copy(self.bin_population[i + 1, point1:d])
                new_pop[i, d:point2] = copy.copy(self.bin_population[i, d:point2])
                new_pop[i, point2:c] = copy.copy(self.bin_population[i + 1, point2:c])
                new_pop[i + 1, 0:point1] = copy.copy(self.bin_population[i + 1, 0:point1])
                new_pop[i + 1, point1:d] = copy.copy(self.bin_population[i, point1:d])
                new_pop[i + 1, d:point2] = copy.copy(self.bin_population[i + 1, d:point2])
                new_pop[i + 1, point2:c] = copy.copy(self.bin_population[i, point2:c])
        self.bin_population = new_pop

    # 种群个体进行变异,其中pm是概率阈值，由他决定是否进行变异
    def mutation(self):
        e = self.chromosome_length
        f1 = int(e / 2)
        random.shuffle(self.bin_population)               #将种群打乱排序
        for i in range(15):                               #选取前10个进行变异
            point3 = random.randint(0, f1 - 1)
            point4 = random.randint(f1, e - 1)
            self.bin_population[i, point3] = (self.bin_population[i, point3] + 1) % 2  # 将该点的0变成1，1变成0
            self.bin_population[i, point4] = (self.bin_population[i, point4] + 1) % 2

if __name__ == '__main__':
    x0 = []  # 设置横坐标点
    y0 = []  # 设置纵坐标点
    x_total = []
    y_total = []
    f_total = []
    p = Population(pop_num = 50, chromosome_length = 36, left=-10, right=10)   #设种群的数目为50
    for i in range(200):
        h = p.function()
        fitness = h[0]  # 得到每一个个体的适应度
        pop1 = h[1]  # 得到X值的数组
        pop2 = h[2]  # 得到Y值的数组
        x0.append(pop1[np.argmax(fitness)])
        y0.append(pop2[np.argmax(fitness)])
        p.selection()
        p.crossover(pc=0.1)  # 进行交叉操作,如果概率大于0.1，则进行交叉
        p.mutation()  # 进行变异操作
        x_index = pop1[np.argmax(fitness)]
        y_index = pop2[np.argmax(fitness)]
        f_max = max(fitness)
        x_total.append(x_index)     #得到该次迭代的取得最大值时X的取值
        y_total.append(y_index)
        f_total.append(f_max)
    f_total = np.array(f_total)
    z_max = np.max(f_total)
    die_num = np.argmax(f_total)
    x_index = x_total[np.argmax(f_total)]
    y_index = y_total[np.argmax(f_total)]
    print(x_index)
    print(die_num)
    print(y_index)
    print(z_max)
    num = []
    for i in range(0, 200, 1):
        num.append(i)
    plt.figure(1)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(num, x_total)
    ax1.set_xlabel("iterations", fontsize = 22)
    ax1.set_ylabel("x_xalue", fontsize = 22)
    ax1.set_title("The value of x in the iteration", fontsize = 22)
    ax1.set_ylim(-2.000, 2.000)
    plt.tick_params(labelsize = 18)
    ax1.plot(die_num, x_index, '*', color='k')
    ax1.text(die_num, x_index + 0.5, str(x_index), ha = 'center', va = 'bottom', fontsize = 22)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(num, y_total)
    ax2.set_xlabel("iterations", fontsize = 22)
    ax2.set_ylabel("y_value",fontsize = 22)
    ax2.set_title("The value of y in the iteration",fontsize = 22)
    ax2.set_ylim(-2.000, 2.000)
    plt.tick_params(labelsize = 18)
    ax2.plot(die_num, y_index, '*', color='k')
    ax2.text(die_num, y_index + 0.5, str(y_index), ha='center', va='bottom', fontsize=22)
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(num, f_total)
    ax3.set_xlabel("iterations", fontsize = 22)
    ax3.set_ylabel("the value of f1", fontsize = 22)
    ax3.set_title("The value of f1 in the iteration", fontsize = 22)
    ax3.set_ylim(0.000, 1.500)
    plt.tick_params(labelsize = 18)
    ax3.plot(die_num, z_max, '*', color='k')
    ax3.text(die_num, z_max + 0.1, str(z_max), ha='center', va='bottom', fontsize=22)
    plt.show()
    x = np.arange(-10., 10., 0.1)
    y = np.arange(-10., 10., 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.sin(Y) / X / Y
    fig = plt.figure(2)
    ax4 = Axes3D(fig)
    ax4.set_xlabel('the value of x',  fontsize = 18)
    ax4.set_ylabel('the value of y',  fontsize = 18)
    ax4.set_zlabel('the value of f',  fontsize = 18)
    ax4.set_title('the functional image(the black dot represents the maximum value)',  fontsize = 18)
    ax4.scatter(x_index, y_index, z_max, c='k', marker='o')
    # plt.plot(x_index,y_index,'*',color = 'k')
    ax4.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()