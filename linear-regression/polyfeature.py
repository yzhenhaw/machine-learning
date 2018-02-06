# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:16:02 2018

@author: zhenyang
"""
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def solveclose(X_train, y_train):#function of Closed Form
    err=0
    w=np.random.uniform(-1,1,X_train.shape[1])#weight
    w=np.dot(np.linalg.inv((np.dot(X_train.transpose(),X_train))),np.dot(X_train.transpose(),y_train))
    
    #for poly=0
#    w=np.random.uniform(-1,1,1)
#    w=np.dot(1/(np.dot(X_train.transpose(),X_train)),np.dot(X_train.transpose(),y_train))
    
    for i in range (0,len(y_train)):
        err=err+(np.dot(w,X_train[i])-y_train[i])**2
    N=X_train.shape[0]
    err=err/N
    err=err**0.5        
    return w,err

def test(w,X_test,y_test):#calculate the testing error
    err=0#MSE
    for i in range (0,len(y_test)):
        err=err+(np.dot(w,X_test[i])-y_test[i])**2
    N=X_test.shape[0]
    err=err/N
    err=err**0.5#RMSE
    return err

def poly(X_train,X_test,j):#function to add poly features
    lentrain=len(X_train)#feature length
    lentest=len(X_test)
    for i in range(0,lentrain):
        if j==2:
            X_train[i]=np.append(X_train[i],X_train[i]**2)
        elif (j==3):
            X_train[i]=np.append(X_train[i],X_train[i]**2)
            X_train[i]=np.append(X_train[i],X_train[i]**3)
        elif j==4:
            X_train[i]=np.append(X_train[i],X_train[i]**2)
            X_train[i]=np.append(X_train[i],X_train[i]**3)
            X_train[i]=np.append(X_train[i],X_train[i]**4)
            
    for ii in range(0,lentest):
        if j==2:
            X_test[ii]=np.append(X_test[ii],X_test[ii]**2)
        elif (j==3):
            X_test[ii]=np.append(X_test[ii],X_test[ii]**2)
            X_test[ii]=np.append(X_test[ii],X_test[ii]**3)
        elif j==4:
            X_test[ii]=np.append(X_test[ii],X_test[ii]**2)
            X_test[ii]=np.append(X_test[ii],X_test[ii]**3)
            X_test[ii]=np.append(X_test[ii],X_test[ii]**4)
            
    return X_train,X_test

def setsize(X_train,y_train,j): #function of adjustment of training set
    length=round(len(X_train)*j)
    XX_train=[]
    yy_train=[]
    for i in range(0,length):
        XX_train.append(X_train[i])
        yy_train.append(y_train[i])
    return XX_train,yy_train

# Load dataset
dataset = datasets.load_boston()

# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []

for k in range(10):

    # Shuffle data
    rand_perm = np.random.permutation(Ndata)
    features = [features_orig[ind] for ind in rand_perm]
    labels = [labels_orig[ind] for ind in rand_perm]

    # Train/test split
    #Nsplit = np.random.randint(50,400)
    Nsplit =50
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    
    #poly 0

#    for i in range (0,len(X_train)):
#        X_train[i]=y_train[i]**0
#    for i in range (0,len(X_test)):
#        X_test[i]=y_test[i]**0

    #polybnomial featrues of 2,3,4
    
    #X_train,X_test=poly(X_train,X_test,4)
    
    #setsize adjustment
    X_train,y_train=setsize(X_train,y_train,1)
    
    # Preprocess your data - Normalization, adding a constant feature, when poly=0, don't normalize it
    trainmean=np.mean(X_train,axis=0)
    trainstd=np.std(X_train,axis=0)
    
    for i in range(0,len(X_train)):
        X_train[i]=(X_train[i]-trainmean)/trainstd
    for i in range(0,len(X_test)):
        X_test[i]=(X_test[i]-trainmean)/trainstd
    
    
    for i in range(0,len(X_train)):
        X_train[i]=np.append(X_train[i],1)
    for i in range(0,len(X_test)):   
        X_test[i]=np.append(X_test[i],1)
    X_train=np.array(X_train)
    X_test=np.array(X_test)


    # Solve for optimal w
    # Use your solver function
    w,error = solveclose(X_train, y_train)
    errtest = test (w,X_test,y_test)
    plt.plot(error)
    plt.ylabel('MSE')
    plt.xlabel('number of epochs')
    plt.show()
    
    np.set_printoptions(threshold=20)#print all the data
    print ('w: ', w)
    print ('testerror: ', errtest)
    print ('trainerror: ', error)

    # Collect train and test errors
    # Use your implementation of the mse function
    train_errs.append(error)
    test_errs.append(errtest)

    print('Mean training error: ', np.mean(train_errs))
    print('Mean test error: ', np.mean(test_errs))
    
    #plot for polynomial features
#    trainpoly=[9.17,21.43,5.783,3.439,3.007]
#    testpoly=[9.36,16.51,5.655,4.323,17.833]
#    x=np.linspace(0,4,5)
#    plt.plot(x,trainpoly,label='train error')
#    plt.plot(x,testpoly,label='test error')
#    plt.legend()
#    plt.xlabel('polynomial features of order')
#    plt.ylabel('RMSE')

    
    #plot for set size change
    trainsize=[4.337,4.475,4.558,4.584,4.685]
    testsize=[4.871,4.886,5.251,5.113,4.629]
    x=np.linspace(0.2,1,5)
    plt.plot(x,trainsize,label='train error')
    plt.plot(x,testsize,label='test error')
    plt.legend()
    plt.xlabel('training set size')
    plt.ylabel('RMSE')
    