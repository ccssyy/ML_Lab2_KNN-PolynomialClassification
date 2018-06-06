import copy
from matplotlib import pyplot as pl
from matplotlib import animation as ani
import csv
import numpy as np
import random

w = [0,0,0,0,0,0,0,0]  # weight vector
b = 0  # bias
yita = 0.5  # learning rate
data = [[(1,4),1],[(0.5,2),1],[(2,2.3),1],[(1,0.5),-
1],[(2,1),-1],[(4,1),-1],[(3.5,4),1],[(3,2.2),-1]]
# data=[[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
record = []


def loadCsv(filename):
    lines = csv.reader(open(filename,"r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

dataset = loadCsv('pima-indians-diabetes.data.csv')
print(len(dataset[0]))
'''
if y(wx+b)<=0,return false; else, return true
'''


def sign(vec):
    global w,b
    res = 0
    wx = 0
    for i in range(len(vec)-1):
        wx += w[i]*vec[i]
    res = vec[-1] * (wx + b)
    if res > 0:
        return 1
    else:
        return -1


'''
update the paramaters w&b
'''


def update(vec):
    global w,b,record
    for i in range(len(vec)-1):
        w[i] = w[i] + yita * vec[-1] * vec[i]
    b = b + yita * vec[-1]
    record.append([copy.copy(w),b])


'''
check and calculate the whole data one time
if all of the input data can be classfied correctly at one time, 
we get the answer
'''


def perceptron(data):
    count = 1
    for ele in data:
        flag = sign(ele)
        if not flag > 0:
            count = 1
            update(ele)
        else:
            count += 1
    if count >= len(data):
        return 1
    else:
        return 0

def Accuracy(result,testset):
    correct = 0
    for instance in testset:
        pred = 0
        for i in range(len(result[0])):
            pred += result[0][i] * instance[i]
        pred += result[1]
        if pred > 0:
            y_pred = 1
        else:
            y_pred = 0
        if y_pred == instance[-1]:
            correct += 1
    return correct/len(testset) * 100


if __name__ == '__main__':
    j=1
    while True:
        j += 1
        shuffle_data = random.shuffle(dataset)
        trainset,testset = dataset[:int(len(dataset) * 0.67)],dataset[int(len(dataset) * 0.67):]
        perceptron(trainset)
        print('step: {}'.format(j))
        print(record[-1])
        acc = Accuracy(record[-1],dataset)
        print('{0}%'.format(acc))
        if acc > 80:
            break
    print('weight is:{0}'.format(record[-1]))
    print('Accuracy: {0}%'.format(acc))
'''
x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []
x5 = []
y5 = []
x6 = []
y6 = []
x7 = []
y7 = []
x8 = []
y8 = []

# display the animation of the line change in searching process
fig = pl.figure()
ax = pl.axes(xlim=(-1,5),ylim=(-1,5))
line, = ax.plot([],[],'g',lw=2)


def init():
    line.set_data([],[])
    for p in dataset:
        if p[len(p)-1] > 0:
            x1.append(p[0])
            x2.append(p[1])
            x3.append(p[2])
            x4.append(p[3])
            x5.append(p[4])
            x6.append(p[5])
            x7.append(p[6])
            x8.append(p[7])
        else:
            y1.append(p[0])
            y2.append(p[1])
            y3.append(p[2])
            y4.append(p[3])
            y5.append(p[4])
            y6.append(p[5])
            y7.append(p[6])
            y8.append(p[7])
    pl.plot(x1,x2,x3,x4,x5,x6,x7,x8,'or')
    pl.plot(y1,y2,y3,y4,y5,y6,y7,y8,'ob')
    return line,


def animate(i):
    global record,ax,line
    w = record[i][0]
    b = record[i][1]
    x1 = -5
    y1 = -(b + w[0] * x1) / w[1]
    x2 = 6
    y2 = -(b + w[0] * x2) / w[1]
    line.set_data([x1,x2],[y1,y2])
    return line,


animat = ani.FuncAnimation(fig,animate,init_func=init,frames=len(record),interval=1000,repeat=True,blit=True)
pl.show()
animat.save('E:/USTC_SSE_COURSES/Machine_Learning/ML_Lab2_KNN-PolynomialClassification/perceptron.gif',fps=2,
            writer='imagemagick')
'''