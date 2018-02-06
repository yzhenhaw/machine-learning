# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:31:17 2018

@author: zhenyang
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def hyper(X_train, y_train, lamda):#function of hyperparameter form
    w=np.random.uniform(-1,1,X_train.shape[1])#weight
    err=0#MSE
    N=X_train.shape[0]#number of input data
    w=np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)+N*lamda*np.identity(X_train.shape[1])),np.dot(X_train.transpose(),y_train))
    for i in range (0,len(y_train)):
        err=err+(np.dot(w,X_train[i])-y_train[i])**2
    N=X_train.shape[0]
    err=err/N#MSE
    err=err**0.5  #RMSE      
    return w,err



def test(w,X_test,y_test):#calculate the testing error and validation error
    err=0#MSE
    for i in range (0,len(y_test)):
        err=err+(np.dot(w,X_test[i])-y_test[i])**2
    N=X_test.shape[0]
    err=err/N#MSE
    err=err**0.5#RMSE
    return err
    
def setsize(X_train,y_train,j): #function of adjustment of training set
    length=round(len(X_train)*j)
    XX_train=[]
    yy_train=[]
    X_valid=[]
    y_valid=[]
    for i in range(0,length):
        XX_train.append(X_train[i])
        yy_train.append(y_train[i])
    for k in range(0,len(X_train)-length):
        X_valid.append(X_train[len(X_train)-k-1])
        y_valid.append(y_train[len(y_train)-k-1])
    return XX_train,yy_train, X_valid, y_valid

# Load dataset
dataset = datasets.load_boston()

# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []
valid_errs=[]

for k in range(100):

    # Shuffle data
    rand_perm = np.random.permutation(Ndata)
    features = [features_orig[ind] for ind in rand_perm]
    labels = [labels_orig[ind] for ind in rand_perm]

    # Train/test split
    #Nsplit = np.random.randint(50,400)
    Nsplit =50
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    
    trainmean=np.mean(X_train,axis=0)
    trainstd=np.std(X_train,axis=0)

    #setsize adjustment
    X_train,y_train,X_valid,y_valid=setsize(X_train,y_train,0.9)
    # Preprocess your data - Normalization, adding a constant feature

    
#    for i in range(0,len(X_train)):
#        X_train[i]=(X_train[i]-trainmean)/trainstd
#    for i in range(0,len(X_test)):
#        X_test[i]=(X_test[i]-trainmean)/trainstd
#    for i in range(0,len(X_valid)):
#        X_valid[i]=(X_valid[i]-trainmean)/trainstd
    
    for i in range(0,len(X_train)):
        X_train[i]=np.append(X_train[i],1)
    for i in range(0,len(X_test)):   
        X_test[i]=np.append(X_test[i],1)
    for i in range(0,len(X_valid)):   
        X_valid[i]=np.append(X_valid[i],1)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    X_valid=np.array(X_valid)


    # Solve for optimal w
    # Use your solver function
    w,error = hyper(X_train, y_train,0)
    
    errvalid=test(w,X_valid,y_valid)
    errtest = test (w,X_test,y_test)
    
    '''
    plt.plot(error)
    plt.ylabel('MSE')
    plt.xlabel('number of epochs')
    plt.show()
    
    np.set_printoptions(threshold=20)#print all the data
    print ('w: ', w)
    print ('testerror: ', errtest)
    print ('trainerror: ', error)
    '''
    # Collect train and test errors
    # Use your implementation of the mse function
    train_errs.append(error)
    test_errs.append(errtest)
    valid_errs.append(errvalid)
    print('Mean training error: ', np.mean(train_errs))
    print('Mean test error: ', np.mean(test_errs))
    print('Mean validation error: ', np.mean(valid_errs))
    
    valid=[5.656,5.117,5.267,4.978,5.404,5.518]
    test=[4.91,5.13,5.083,5.253,5.443,5.475]
    x=np.linspace(0,0.5,6)
    plt.plot(x,valid,label='valid error')
    plt.plot(x,test,label='test error')
    plt.legend()
    plt.xlabel('hyperparameter')
    plt.ylabel('RMSE')