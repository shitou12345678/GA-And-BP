import math
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#定义活化函数
def Sigmoid(x):
    return np.tanh(x)
#定义活化函数的导数
def Sigmoid_Derivative(x):
    return 1-np.tanh(x)*np.tanh(x)

class NN(object):
    # 第一步：初始化，设置两个隐藏层
    def __init__(self, input_layer_num, hidden1_layer_num, hidden2_layer_num, output_layer_num, learning_rate):
        self.input_n = input_layer_num                                  #输入层的神经元个数
        self.hidden1_n = hidden1_layer_num                              #第一个隐藏层神经元的个数
        self.hidden2_n = hidden2_layer_num                              #第二个隐藏层神经元的个数
        self.output_n = output_layer_num                                #输出层的神经元个数
        self.rate = learning_rate                                       #设置学习率
        # 初始化各个层的输出结果，不设置偏值项
        self.input_mat = np.zeros(self.input_n)        #初始化输入层的数组
        self.hidden1_mat = np.zeros(self.hidden1_n)    #初始化第一个隐藏层的数组,self.hidden1表示未归一化的值
        self.hidden1 = np.zeros(self.hidden1_n)
        self.hidden2_mat = np.zeros(self.hidden2_n)    #初始化第二个隐藏层的数组，self.hidden2表示未归一化的值
        self.hidden2 = np.zeros(self.hidden2_n)
        self.output_mat = np.zeros(self.output_n)      #初始化输出层的数组,self.output表示未归一化的值
        self.output = np.zeros(self.output_n)
        #初始化各个层之间的权重
        self.input_hidden1_weights = np.empty([self.input_n, self.hidden1_n])       #初始化输入层与第一个隐藏层之间的权重
        self.hidden1_hidden2_weights = np.empty([self.hidden1_n, self.hidden2_n])  #初始化第一个隐藏层与第二个隐藏层之间的权重
        self.hidden2_output_weights = np.empty([self.hidden2_n, self.output_n])    #初始化第二个隐藏层与输出层之间的权重
        #初始化各个层之间的误差
        self.output_error = np.zeros(self.output_n)       #初始化输出层的误差
        self.hidden2_error = np.zeros(self.hidden2_n)     #初始化第二个隐藏层的误差
        self.hidden1_error = np.zeros(self.hidden1_n)     #初始化第一个隐藏层的误差
        self.input_erroe = np.zeros(self.input_n)         #初始化输入层的误差，一般不计算输入层的误差
        random.seed(1)                                    #引入seed()函数让每一次的权重随机化都是一样的，即保证每一次第一次的权重都是相同的
        for i in range(self.input_n):
            for j in range(self.hidden1_n):
                self.input_hidden1_weights[i][j] = random.random()       #随机得到输入和第一层隐藏层的权重
        random.seed(2)
        for i in range(self.hidden1_n):
            for j in range(self.hidden2_n):
                self.hidden1_hidden2_weights[i][j] = random.random()     #随机得到第一层隐藏层和第二层隐藏层的权重
        random.seed(3)
        for i in range(self.hidden2_n):
            for j in range(self.output_n):
                self.hidden2_output_weights[i][j] = random.random()      #随机得到第二层隐藏层与输出层的权重

    #第二步：前向计算,得到输入对应的输出值
    def Forward_Calculation(self, sample):
        for i in range(len(self.input_mat)):
            self.input_mat[i] = sample
        for i in range(self.hidden1_n):
            hidden1_output = 0
            for j in range(self.input_n):
                hidden1_output += self.input_mat[j] * self.input_hidden1_weights[j][i]        #计算第一层隐含层的输出
            self.hidden1[i] = hidden1_output                                                 #第一层隐藏层输出的数组
            self.hidden1_mat[i] = Sigmoid(hidden1_output)                                #计算输出的活化函数值
        for i in range(self.hidden2_n):
            hidden2_output = 0
            for j in range(self.hidden1_n):
                hidden2_output += self.hidden1_mat[j] * self.hidden1_hidden2_weights[j][i]   #计算第二层隐含层的输出
            self.hidden2[i] = hidden2_output
            self.hidden2_mat[i] = Sigmoid(hidden2_output)                                #计算其输出活化函数值
        for i in range(self.output_n):
            output = 0
            for j in range(self.hidden2_n):
                output += self.hidden2_mat[j] * self.hidden2_output_weights[j][i]          #计算输出输出层的输出
            self.output[i] = output
            self.output_mat[i] = Sigmoid(output)                                        #计算输出层的活化函数值
        return output                                                                          #返回得到的输出值

    #第三步：反向计算
    def Reverse_Calculation(self, label):
        for i in range(self.output_n):
            self.output_error[i] = (label - self.output_mat[i]) * Sigmoid_Derivative(self.output[i])    #计算输出层的误差
        for i in range(self.hidden2_n):
            a = 0
            for j in range(self.output_n):
                a += self.hidden2_output_weights[i][j] * self.output_error[j]
            self.hidden2_error[i] = a * Sigmoid_Derivative(self.hidden2[i])          #得到第二层隐藏层的误差
        for i in range(self.hidden1_n):
            b = 0
            for j in range(self.hidden2_n):
                b += self.hidden1_hidden2_weights[i][j] * self.hidden2_error[j]
            self.hidden1_error[i] = b * Sigmoid_Derivative(self.hidden1[i])            #得到第一层隐藏层的误差
        E_q = 0
        for i in range(self.output_n):
            E_q += 0.5 * ((label - self.output[i]) ** 2)                                #计算样本的能量函数
        return E_q                                                                         #返回能量函数

    #第四步：修正权重
    def correct_weight(self):
        for i in range(self.hidden2_n):
            for j in range(self.output_n):
                #修正第二层隐藏层的权值
                self.hidden2_output_weights[i][j] = copy.copy(self.hidden2_output_weights[i][j]) + self.output_error[j] * self.rate * self.hidden2_mat[i]
        for i in range(self.hidden1_n):
            for j in range(self.hidden2_n):
                #修正第一层隐藏层的权值
                self.hidden1_hidden2_weights[i][j] = copy.copy(self.hidden1_hidden2_weights[i][j]) + self.hidden2_error[j] *self.rate * self.hidden1_mat[i]
        for i in range(self.input_n):
            for j in range(self.hidden1_n):
                #修正输入层的权重
                self.input_hidden1_weights[i][j] = copy.copy(self.input_hidden1_weights[i][j]) + self.hidden1_error[j] * self.rate * self.input_mat[i]

    #训练样本值
    def train(self, samples_train, labels_train, iterations):
        E_train_mat = np.zeros(iterations)     #用来存放每一次迭代的误差
        output_train = np.zeros(len(samples_train))    #用来存放最终训练后实际的值
        for i in range(iterations):
            E_train = 0
            for j in range(len(samples_train)):
                sample_train = samples_train[j]
                label_train = labels_train[j]
                output_train[j] = self.Forward_Calculation(sample_train)       #得到训练的实际值
                E_train += self.Reverse_Calculation(label_train)
                self.correct_weight()
            E_train_mat[i] = E_train                     #这是每一次迭代次数的误差
        return [E_train_mat, output_train]               #返回最后一次迭代的实际值和每次迭代的误差

    #测试样本值
    def test(self, samples_test, labels_test):
        E_test = 0
        output_test = np.zeros(len(samples_test))
        for i in range(len(samples_test)):
            sample_test = samples_test[i]
            label_test = labels_test[i]
            output_test[i] = self.Forward_Calculation(sample_test)
            E_test += self.Reverse_Calculation(label_test)             #得到检验误差
        return [E_test, output_test]                                   #返回检验误差和检验得到的实际值

if __name__ == '__main__':
    samples_train_n = 9               #训练样本所需的样本数
    samples_test_n = 361              #检验样本所需的样本数
    samples_train = np.empty(samples_train_n)          #建立一个训练样本集和其对应期望的值
    labels_train = np.empty(samples_train_n)
    samples_test = np.empty(samples_test_n)           #建立一个检验样本集和其对应期望的值
    labels_test = np.empty(samples_test_n)
    for i in range(samples_train_n):
        samples_train[i] = 1.0 * i * 2 * math.pi / samples_train_n            #在0到2π之间等距离取9个值作为训练样本
        labels_train[i] = np.sin(samples_train[i])                            #计算训练样本值所对应的期望值
    for i in range(samples_test_n):
        samples_test[i] = 1.0 * i *2 * math.pi / samples_test_n               #在0到2π之间等距离取361个值作为检验样本
        labels_test[i] = np.sin(samples_test[i])                              #计算检验样本值所对应的期望值
    nn = NN(input_layer_num=1, hidden1_layer_num=10, hidden2_layer_num=10, output_layer_num=1, learning_rate=0.05)
    E_train_mat = nn.train(samples_train, labels_train, iterations=10000)[0]
    output_train = nn.train(samples_train, labels_train, iterations=10000)[1]
    E_test = nn.test(samples_test, labels_test)[0]
    output_test = nn.test(samples_test, labels_test)[1]
    print(output_test)
    print(E_test)
    num = []
    for i in range(len(E_train_mat)):
        num.append(i)
    die_num = 1000                                                #取最终迭代次数
    fina_error = E_train_mat[die_num - 1]                             #取最终的训练误差
    plt.figure(1)                                                 #画第一张图，训练误差随训练次数的变化图
    plt.plot(num, E_train_mat)
    plt.ylim(-2.000, 2.000)
    plt.xlabel("iterations", fontsize=22)
    plt.ylabel("error", fontsize=22)
    plt.title("The error in the iteration", fontsize=22)
    plt.tick_params(labelsize=18)
    plt.plot(die_num, fina_error, '*', color='k')             #在图中标出最终的训练误差
    plt.text(die_num, fina_error + 0.5, str(fina_error), ha='center', va='bottom', fontsize=22)
    plt.show()
    #画第二张图，训练函数图和检验函数图
    num2 = []
    for i in range(len(output_train)):
        num2.append(i)
    num3 = []
    for i in range(len(output_test)):
        num3.append(i)
    plt.figure(2)
    ax1 = plt.subplot(211)              #画第一张图，训练函数图
    ax1.plot(num2, output_train)        #训练样本值和实际输出值
    ax1.plot(num2, labels_train)         #训练样本图和期望输出值
    ax1.set_xlabel("Training Sample", fontsize = 22)
    ax1.set_ylabel("Train value/ Expected value", fontsize = 22)
    ax1.set_title("Actual/Expected Training Sample Values", fontsize = 22)
    ax1.set_ylim(-2.000, 2.000)
    ax2 = plt.subplot(212)                  #画第二张图，检验样本图
    ax2.plot(num3, output_test)             #测试样本值和实际输出值
    ax2.plot(num3, labels_test)             #测试样本值和期望输出值
    ax2.set_xlabel("Testing Sample", fontsize = 22)
    ax2.set_ylabel("Testing value/ Expected value", fontsize = 22)
    ax2.set_title("Actual/Expected Testing Sample Values", fontsize = 22)
    ax2.set_ylim(-2.000, 2.000)
    plt.show()




