import numpy as np
import matplotlib.pyplot as plt
#dataset a
#data = np.zeros((100, 3))
#val = np.random.uniform(0, 2, 100)
#diff = np.random.uniform(-1, 1, 100)
#data[:,0], data[:,1], data[:,2] = val - diff, val + diff, np.ones(100)
#target = np.asarray(val > 1, dtype = int) * 2 - 1

#dataset b
data = np.ones((100, 3))
data[:50,0], data[50:,0] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
data[:50,1], data[50:,1] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
target = np.zeros(100)
target[:50], target[50:] = -1 * np.ones(50), np.ones(50)

def solveclose(X_train, y_train):#function of Closed Form
    w=np.zeros(3)#weight
    w=np.dot(np.linalg.inv((np.dot(X_train.transpose(),X_train))),np.dot(X_train.transpose(),y_train))
    return w

w=np.zeros(3)#weight intialization

#perceptron
for i in range(0,10):
    for j in range(0,data.shape[0]):        
        if(target[j]*np.dot(w,data[j])>0):
            w=w;
        else:
            w=w+target[j]*data[j]

fig1=plt.figure()
plt.scatter(data[:,0],data[:,1],c=target)
plt.plot(data[:,0],(-w[0]*data[:,0]-w[2])/w[1])#w1x1+w2x2+w3=0


#intialize w with linear regression, and then do perceptron
w=solveclose(data, target)
#perceptron
for i in range(0,10):
    for j in range(0,data.shape[0]):        
        if(target[j]*np.dot(w,data[j])>0):
            w=w;
        else:
            w=w+target[j]*data[j]

fig2=plt.figure()
plt.scatter(data[:,0],data[:,1],c=target)
plt.plot(data[:,0],(-w[0]*data[:,0]-w[2])/w[1])#w1x1+w2x2+w3=0