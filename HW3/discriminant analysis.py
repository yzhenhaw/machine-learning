import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()

# You have two features and two classifications
data_0, data_1 = iris.data[:,1:3][:50], iris.data[:,1:3][50:100]

# TODO: Compute the mean and covariance of each cluster, and use these to find a QDA Boundary
data_0_mean=np.mean(data_0,axis=0)
data_1_mean=np.mean(data_1,axis=0)
data_0_cov=np.cov(data_0.transpose())
data_1_cov=np.cov(data_1.transpose())
data=np.concatenate((data_0, data_1), axis=0)
data_mean=np.mean(data,axis=0)
data_cov=np.cov(data.transpose())
data_covinv=np.linalg.inv(data_cov)
data_0_covinv=np.linalg.inv(data_0_cov)
data_1_covinv=np.linalg.inv(data_1_cov)
print("datatotalmean",data_mean)
print("datatotalcovariance",data_cov)
print("data0mean",data_0_mean)
print("data0covariance",data_0_cov)
print("data1mean",data_1_mean)
print("data1covariance",data_1_cov)

def LDA(x):
    z=np.dot(x,np.dot((data_1_mean-data_0_mean).transpose(),data_covinv))+np.log(5/5)-0.5*np.dot(np.dot(data_1_mean.transpose(),data_covinv),data_1_mean)+0.5*np.dot(np.dot(data_0_mean.transpose(),data_covinv),data_0_mean)
    return z

def QDA(x):
    z=0.5*(x-data_1_mean).transpose().dot(data_1_covinv).dot(x-data_1_mean)-0.5*(x-data_0_mean).transpose().dot(data_0_covinv).dot(x-data_0_mean)-np.log(5/5)+0.5*np.log(np.linalg.det(data_1_cov)/np.linalg.det(data_0_cov))
    return z
#0.5*np.dot(np.dot(x.transpose(),data_1_stdinv-data_0_stdinv),x)-2*np.dot((np.dot(data_0_mean.transpose(),data_0_stdinv)+np.dot(data_1_mean.transpose(),data_1_stdinv)),x)

def plot_contour_LDA():
    plot_delta_x = (np.max(data[:,0])-np.min(data[:,0]))/100
    plot_delta_y = (np.max(data[:,1])-np.min(data[:,1]))/100
    plot_x = np.arange(np.min(data[:,0]), np.max(data[:,0]), plot_delta_x)
    plot_y = np.arange(np.min(data[:,1]), np.max(data[:,1]), plot_delta_y)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[i][j] = LDA(np.array([X[i][j],Y[i][j]]))
            
    cs = plt.contour(X, Y, Z, levels=[0])
    #plt.clabel(cs, inline=0.1, fontsize=10)
    axes = plt.gca()
    axes.set_xlim([np.min(data[:,0]), np.max(data[:,0])])
    axes.set_ylim([np.min(data[:,1]), np.max(data[:,1])])
    plt.show()
    
def plot_contour_QDA():
    plot_delta_x = (np.max(data[:,0])-np.min(data[:,0]))/100
    plot_delta_y = (np.max(data[:,1])-np.min(data[:,1]))/100
    plot_x = np.arange(np.min(data[:,0]), np.max(data[:,0]), plot_delta_x)
    plot_y = np.arange(np.min(data[:,1]), np.max(data[:,1]), plot_delta_y)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[i][j] = QDA(np.array([X[i][j],Y[i][j]]))
            
    cs = plt.contour(X, Y, Z, levels=[0])
    #plt.clabel(cs, inline=0.1, fontsize=10)
    axes = plt.gca()
    axes.set_xlim([np.min(data[:,0]), np.max(data[:,0])])
    axes.set_ylim([np.min(data[:,1]), np.max(data[:,1])])
    plt.show()
    
fig1=plt.figure()
plt.scatter(data_0[:,0],data_0[:,1])    
plt.scatter(data_1[:,0],data_1[:,1])    
plot_contour_LDA()

fig2=plt.figure()
plt.scatter(data_0[:,0],data_0[:,1])    
plt.scatter(data_1[:,0],data_1[:,1])    
plot_contour_QDA()
# TODO: Compute the mean and covariance of the entire dataset, and use these to find a LDA Boundary

# TODO: Make two scatterplots of the data, one showing the QDA Boundary and one showing the LDA Boundary