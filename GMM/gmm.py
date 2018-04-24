import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

def random_posdef(n):
  A = np.random.rand(n, n)
  return np.dot(A, A.transpose())

# Parameter initialization ###
K = 3
pi = [1.0/K for i in range(K)]
means = [[0,0] for i in range(K)]
covs = [random_posdef(2) for i in range(K)]
##############################
PI=3.1415926
data=np.load('gmm_data.npy')

def Gaussian(x,mu,sig):
    return (1/((2*PI)**2*np.linalg.det(sig))**0.5)*np.exp(-0.5*np.dot(np.dot((x-mu).transpose(),np.linalg.inv(sig)),(x-mu)))

N=data.shape[0]

def plot_contour(mean, sig):
    plot_delta = 0.025
    plot_x = plot_y = np.arange(np.min(np.min(data)), np.max(np.max(data)), plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(len(X)):
        for j in range(len(Y)):
            xmu=np.array([X[i][j]-mean[0],Y[i][j]-mean[1]])
            Z[i][j] = (1/((2*PI)**2*np.linalg.det(sig))**0.5)*np.exp(-0.5*np.dot(np.dot((xmu).transpose(),np.linalg.inv(sig)),(xmu)))
    plt.contour(X, Y, Z, levels=[0.01],colors='k',linewidths=3)

iteration=[1,5,10,20,50]
for i in range(len(iteration)):
    epochnum=iteration[i]
    for epoch in range(epochnum):
        r=np.zeros((N,K))
        for i in range(0,N):
            rsum=0
            for m in range(0,K):
                rsum+=pi[m]*Gaussian(data[i],means[m],covs[m])
            for m in range(0,K):
                r[i,m]=pi[m]*Gaussian(data[i],means[m],covs[m])/rsum
                
        Num=np.sum(r,axis=0)
        covs = [np.zeros((2,2)) for k in range(K)]
        means = [[0,0] for i in range(K)]
        for m in range(K):
            pi[m]=Num[m]/N
            for i in range(0,N):
                means[m]=means[m]+1/Num[m]*np.dot(r[i,m],data[i])
            for i in range(0,N):
                covs[m]=covs[m]+1/Num[m]*np.outer(np.dot(r[i,m],(data[i]-means[m])),data[i]-means[m])

    
    label=np.zeros(N)
    for i in range(0,N):
        label[i]=np.argmax(r[i,:])
        
    plt.figure(i)    
    plt.scatter(data[:,0],data[:,1],c=label)
    for m in range(K):
        plot_contour(means[m], covs[m])
    plt.show()
#    print("pi",pi)
#    print("means",means)
#    print("covs",covs)
    
    
    
    