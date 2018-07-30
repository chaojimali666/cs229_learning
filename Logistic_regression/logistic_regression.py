import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import *

'''
x=np.array([1,2,3,4])
y=np.array([2,4,6,8])
fig, ax = plt.subplots()
#ax.plot(x, y)
a = [1,2,3,4]
b = [3,4,5,6]
ax.scatter(a, b, c='red', s=30, alpha=0.5)
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
plt.show()
'''

def load_data():
    #从数据集中获取数据
    data = []
    lable = [] 
    fr = open('testSet.txt')
    a = fr.readlines()
    for line in a:
    	row = line.strip().split() # 读取的数据为字符串格式
    	data.append([1,float(row[0]),float(row[1])])
    	lable.append(int(row[2]))
    return data,lable

def sigmoroid(X):
    return 1/(1+exp(-X))

def gradAscent(data,lable):
    '''利用梯度下降法求参数'''
    # 列表转矩阵
    X_training = mat(data)
    lable = mat(lable).transpose()
    print(lable.shape)
    m,n = shape(data)
    #将参数初始化为1
    weights = ones((n,1))
    maxStep = 20000 
    #学习速率为 0.001
    alpha = 0.001
    for i in range(maxStep):
        h_teta = sigmoroid(X_training * weights)    
        errors = lable - h_teta
        weights = weights + alpha*X_training.transpose()*errors
    return weights

def stoch_gradAscent(data,lable):
    '''随机梯度下降法'''
    X_training = mat(data)
    lable = mat(lable).transpose()
    m,n = shape(data)
    #
    weights = ones((n,1))
    maxStep = 1000
    #alpha = 0.001
    w0=[weights[0]]
    w_num = 1
    num = [1]
    for i in range(maxStep):
    	for j in range(m):
    		alpha = 4/(1+i+j) + 0.001
    		Xj = X_training[j]
    		h = sigmoroid(Xj*weights)
    		errors = lable[j] - h
    		weights = weights + alpha*Xj.transpose()*errors
    		w_num = w_num +1
    		w0.append(weights[0])
    		num.append(w_num)
    test=[w0,num]
    return weights,test


def plot_bestFit(wei):
    # 画出拟合后的结果
    #w0 = test[0]
    #num = test[1]

    weights = wei.getA()  #将numpy矩阵转换为数组
    data,lable = load_data()
    train_data = array(data)
    m,n = shape(train_data)
    x_record_0 = []
    y_record_0 = []
    
    x_record_1 = []
    y_record_1 = []
    for i in range(m):
        if lable[i] == 0:
            x_record_0.append(train_data[i][1])
            y_record_0.append(train_data[i][2])
        else:
            x_record_1.append(train_data[i][1])
            y_record_1.append(train_data[i][2])

    #plt.subplot(212)
    #plt.plot(num,w0)

    plt.subplot(211)
    plt.scatter(x_record_0, y_record_0, c='red', s=30, marker='s')
    plt.scatter(x_record_1, y_record_1, c='blue', s=30)

    '''
    fig, ax = plt.subplots()
    ax.scatter(x_record_0, y_record_0, c='red', s=30, marker='s')
    ax.scatter(x_record_1, y_record_1, c='blue', s=30)
	'''

    x = arange(-3,3,0.1)
    y = -(weights[0]+weights[1]*x)/weights[2]
    plt.plot(x,y)

    '''
    ax.set(xlabel='X1', ylabel='X2')
    ax.plot(x, y)
    ax.grid()
	'''
    plt.show()


data,lable = load_data()
wei = gradAscent(data,lable)
print(wei)
plot_bestFit(wei)


