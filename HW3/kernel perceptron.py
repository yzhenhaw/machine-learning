import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)
# data and target are given below 
# data is a numpy array consisting of 100 2-dimensional points
# target is a numpy array consisting of 100 values of 1 or -1
data = np.ones((100, 2))
data[:,0] = np.random.uniform(-1.5, 1.5, 100)
data[:,1] = np.random.uniform(-2, 2, 100)
z = data[:,0] ** 2 + ( data[:,1] - (data[:,0] ** 2) ** 0.333 ) ** 2  
target = np.asarray( z > 1.5, dtype = int) * 2 - 1

sigma = [0.1,1.0]

def kernel(x1,x2,sigma):
    k=np.exp(-(np.linalg.norm(x1-x2) ** 2) / (2*sigma*sigma))
    return k

def yprediction(data,data_j,sigma,target, alpha):
    ypre=np.zeros(data.shape[0])
    for i in range(0, data.shape[0]):
        ypre=ypre+kernel(data[i,:],data_j,sigma[1])*target[i]*alpha[i]#change sigma here!!!!!!!!!!!!!!!!!!!!!
    return ypre

def kernelperceptron(data,sigma,target):
    alpha=np.zeros(len(data))
    for epoch in range(0,10):
        for j in range (0, data.shape[0]):
            ypre=np.sign(yprediction(data,data[j,:],sigma,target, alpha))
            if (ypre[j]!=target[j]):
                alpha[j]=alpha[j]+1
    return alpha

def plot_contour():

    plot_delta = (np.max(data)-np.min(data))/100
    plot_x = np.arange(np.min(data), np.max(data), plot_delta)
    plot_y = np.arange(np.min(data), np.max(data), plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[i][j] = yprediction(data,np.array([X[i,j],Y[i,j]]),sigma,target, alpha)[j]
            

    cs = plt.contour(X, Y, Z, levels=[0])
    #plt.clabel(cs, inline=0.1, fontsize=10)
    axes = plt.gca()
    axes.set_xlim([np.min(data), np.max(data)])
    axes.set_ylim([np.min(data), np.max(data)])
    plt.show()
    
alpha=kernelperceptron(data,sigma,target)
print(alpha)

fig1=plt.figure()
plt.scatter(data[:,0],data[:,1],c=target)
plot_contour()
