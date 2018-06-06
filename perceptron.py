import copy
from matplotlib import pyplot as plt
from matplotlib import animation as ani

w = [0,0]  # weight vector
b = 0      # bias
yita = 0.5 # learning rate

# data=[[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
data=[[(1,4),1],[(0.5,2),1],[(2,2.3), 1], [(1, 0.5), -
1], [(2, 1), -1],[(4,1),-1],[(3.5,4),1],[(3,2.2),-1]]

record = []

'''
if y(wx+b)<=0,return false; else, return true
'''
def sign(vec):
    global w,b
    res = 0
    res = vec[1]*(w[0]*vec[0][0] + w[1]*vec[0][1] + b)
    if res > 0:
        return 1
    else:
        return -1

'''
update the paramaters w&b
'''
def update(vec):
    global w,b,record
    w[0] = w[0] + yita*vec[1]*vec[0][0]
    w[1] = w[1] + yita*vec[1]*vec[0][1]
    b = b + yita*vec[1]
    record.appand([copy.copy(w),b])

'''
check and calculate the whole data one time
if all of the input data can be classfied correctly at one time,
we get the answer
'''

def perceptron():
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

if __name__ == '__main__':
    while 1:
        if perceptron() > 0:
            break
    print(record)


