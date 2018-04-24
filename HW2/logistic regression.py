import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

def sigma(x):
    return 1/(1+np.exp(-x))

def solveSGD(X_train, y_train,X_test,y_test):#function of SGD with logistic regression
    rate=0.02#learning rate
    N=X_train.shape[0]
    w=np.random.uniform(-1,1,X_train.shape[1])#weight  
    errortrain=[] #learning error of each iteration for training set
    errortest=[] #learning error of each iteration for testing set
    merrortrain=[] #mean learning error of each iteration for training set
    merrortest=[] # mean learning error of each iteration for testing set  
    errtrain=0#classification accuracy of training set
    errtest=0#classification accuracy of testing set
    merrtrain=0#mean classification accuracy of training set
    merrtest=0#mean classification accuracy of testing set
    order=np.arange(len(y_train))
    np.random.shuffle(order)
    for k in range (0,len(y_train)):
        i=order[k]
        w=w-rate*(sigma(np.dot(w,X_train[i]))-y_train[i])*X_train[i]
        errtrain=sum(np.abs(np.sign(np.dot(X_train,w))-y_train)/2)/N#classification accuracy of training set
        errtest=sum(np.abs(np.sign(np.dot(X_test,w))-y_test)/2)/N#classification accuracy of training set
        merrtrain=(errtrain*k+sum(np.abs(np.sign(np.dot(X_train,w))-y_train)/2)/N)/(k+1)#mean classification accuracy of training set
        merrtest=(errtrain*k+sum(np.abs(np.sign(np.dot(X_test,w))-y_test)/2)/N)/(k+1)#mean classification accuracy of training set        
        errortrain.append(errtrain)       
        errortest.append(errtest)
        merrortrain.append(merrtrain)       
        merrortest.append(merrtest)
    return w,errortrain,errortest,merrortrain,merrortest

def normalizaion(X_train,X_test):
    trainmean=np.mean(X_train,axis=0)
    trainstd=np.std(X_train,axis=0)
    for i in range(0,len(X_train)):
        X_train[i]=(X_train[i]-trainmean)/trainstd
    for i in range(0,len(X_test)):
        X_test[i]=(X_test[i]-trainmean)/trainstd
    X_train = np.insert(X_train, X_train.shape[1], 1, axis=1)
    X_test = np.insert(X_test, X_test.shape[1], 1, axis=1)
    return X_train, X_test

def crossen(w,x,y):
    return -np.dot(np.log(sigma(np.dot(x,w))),y)-np.dot(np.log(1-sigma(np.dot(x,w))),1-y)

# Normalized Binary dataset
# 4 features, 100 examples, 50 labeled 0 and 50 labeled 1
X, y = load_breast_cancer().data, load_breast_cancer().target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train, X_test=normalizaion(X_train,X_test)

w,errortrain,errortest,merrortrain,merrortest = solveSGD(X_train, y_train,X_test,y_test)
crosstrain=crossen(w,X_train,y_train)
crosstest=crossen(w,X_test,y_test)
np.set_printoptions(threshold=np.nan)
print ('w: ', w)
print ('crosstrain: ', crosstrain)
print ('crosstest: ', crosstest)
print ('final train accuracy: ', errortrain[-1])
print ('final test accuracy: ', errortest[-1])

fig1=plt.figure()
plt.plot(errortrain,label='train error')
plt.plot(errortest,label='test error')
plt.legend()
plt.xlabel('iteration of SGD')
plt.ylabel('error')

fig2=plt.figure()
plt.plot(merrortrain,label='mean train error')
plt.plot(merrortest,label='mean test error')
plt.legend()
plt.xlabel('iteration of SGD')
plt.ylabel('error')