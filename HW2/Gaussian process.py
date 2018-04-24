import numpy as np
import matplotlib.pyplot as plt

sigma = [0.3,0.5,1.0]

def kernel(x1,x2,sigma):
    k=np.exp(-((x1-x2) ** 2) / (2*sigma))#Gaussian kernel
    return k

def covariance(X1,X2):
    co=[[kernel(x1,x2,sigma[1]) for x1 in X1] for x2 in X2]#covariance matrix
    return co

X=np.linspace(-5,5,100)
mean=np.zeros(100)
sig=covariance(X,X)
fig1=plt.figure()
for i in range(0,5):
    Y=np.random.multivariate_normal(mean,sig)
    plt.plot(X,Y)
 
fig2=plt.figure()
# observation data points
xd=[-1.3,2.4,-2.5,-3.3,0.3]
yd=[2,5.2,-1.5,-0.8,0.3]

x=np.linspace(-5,5,100)
meanpo=np.dot(np.dot(covariance(xd,x),np.linalg.inv(covariance(xd,xd))),yd)#posterior mean
sigpo=covariance(x,x)-np.dot(np.dot(covariance(xd,x),np.linalg.inv(covariance(xd,xd))),covariance(x,xd))#posterior covariance
for i in range(0,5):
    Ypo=np.random.multivariate_normal(mean,sig)
    plt.plot(x,Ypo)
plt.scatter(xd,yd,s=100)
plt.plot(x,meanpo,linewidth=5.0)
