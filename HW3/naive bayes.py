import numpy as np


# load the training data 
train_features = np.load("spam_train_features.npy")
train_labels = np.load("spam_train_labels.npy")


# load the test data 
test_features = np.load("spam_test_features.npy")
test_labels = np.load("spam_test_labels.npy")

# pic for label 1 and label 0
N=len(train_labels)
Nc1=len(train_features[train_labels==1])
Nc0=len(train_features[train_labels==0])
pi1=(Nc1+1)/(N+2)
pi0=(Nc0+1)/(N+2)

#theta
Nc1dm1=np.sum(train_features[train_labels==1], axis=0)
Nc1dm0=np.ones(np.sum(train_features[train_labels==1], axis=0).shape[0])*Nc1-np.sum(train_features[train_labels==1], axis=0)
Nc0dm1=np.sum(train_features[train_labels==0], axis=0)
Nc0dm0=np.ones(np.sum(train_features[train_labels==0], axis=0).shape[0])*Nc0-np.sum(train_features[train_labels==0], axis=0)

theta_c1dm1=(Nc1dm1+np.ones(len(Nc1dm1)))/(Nc1+2)
theta_c1dm0=(Nc1dm0+np.ones(len(Nc1dm0)))/(Nc1+2)
theta_c0dm1=(Nc0dm1+np.ones(len(Nc0dm1)))/(Nc0+2)
theta_c0dm0=(Nc0dm0+np.ones(len(Nc0dm0)))/(Nc0+2)

#use log here to make prediction
y1=np.dot(test_features,np.log(theta_c1dm1))-np.dot(test_features-np.ones(test_features.shape[1]),np.log(theta_c1dm0))+np.ones(len(np.dot(test_features,np.log(theta_c1dm1))))*np.log(Nc1)
y0=np.dot(test_features,np.log(theta_c0dm1))-np.dot(test_features-np.ones(test_features.shape[1]),np.log(theta_c0dm0))+np.ones(len(np.dot(test_features,np.log(theta_c0dm0))))*np.log(Nc1)

y=[]
for i in range(0,len(y1)):
    if(y1[i]>y0[i]):
        y.append(1)
    else:
        y.append(0)

print("prediction",y)

error=[]
for i in range(0,len(y)):
    if(y[i]!=test_labels[i]):
        error.append(1)

err=sum(error)/len(y)
print("error",err)