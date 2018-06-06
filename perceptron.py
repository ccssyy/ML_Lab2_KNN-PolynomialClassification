import copy
from matplotlib import pyplot as pl
from matplotlib import animation as ani

w = [0,0]  # weight vector
b = 0  # bias
yita = 0.5  # learning rate
data = [[(1,4),1],[(0.5,2),1],[(2,2.3),1],[(1,0.5),-
1],[(2,1),-1],[(4,1),-1],[(3.5,4),1],[(3,2.2),-1]]
# data=[[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
record = []
'''
if y(wx+b)<=0,return false; else, return true
'''


def sign(vec):
    global w,b
    res = 0
    res = vec[1] * (w[0] * vec[0][0] + w[1] * vec[0][1] + b)
    if res > 0:
        return 1
    else:
        return -1


'''
update the paramaters w&b
'''


def update(vec):
    global w,b,record
    w[0] = w[0] + yita * vec[1] * vec[0][0]
    w[1] = w[1] + yita * vec[1] * vec[0][1]
    b = b + yita * vec[1]
    record.append([copy.copy(w),b])


'''
check and calculate the whole data one time
if all of the input data can be classfied correctly at one time, 
we get the answer
'''

def perceptron():
    count = 1
    for ele in data:
        flag = sign(ele)
        if not flag>0:
            count = 1
            update(ele)
        else:
            count+=1
    if count>=len(data):
        return 1

if __name__ == '__main__':
    while 1:
        if perceptron() > 0:
            break
    print(record)

x1=[]
y1=[]
x2=[]
y2=[]

# display the animation of the line change in searching process
fig = pl.figure()
ax = pl.axes(xlim=(-1,5),ylim=(-1,5))
line,_=ax.plot([],[],'g',lw=2)

def init():
    line.set_data([],[])
    for p in data:
        if p[1]>0:
            x1.append(p[0][0])
            y1.append(p[0][1])
        else:
            x2.append(p[0][0])
            y2.append(p[0][1])
    pl.plot(x1,y1,'or')
    pl.plot(x2,y2,'ob')
    return line,

def animate(i):
    global record,ax,line
    w=record[i][0]
    b=record[i][1]
    x1=-5
    y1=-(b+w[0]*x1)/w[1]
    x2=6
    y2=-(b+w[0]*x2)/w[1]
    line.set_data([x1,x2],[y1,y2])
    return line,

animat = ani.FuncAnimation(fig,animate,init_func=init,frames=len(record),interval=1000,repeat=True,blit=True)
pl.show()
animat.save('E:/USTC_SSE_COURSES/Machine_Learning/ML_Lab2_KNN-PolynomialClassification/perceptron.gif',fps=2,writer='imagemagick')