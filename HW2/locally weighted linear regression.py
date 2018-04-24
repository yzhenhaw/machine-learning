import numpy as np
import matplotlib.pyplot as plt

def errMSE(X_train, y_train,w):#MSE error
    err=0
    for i in range (0,len(y_train)):
        err=err+(np.dot(X_train[i],w)-y_train[i])**2
    N=X_train.shape[0]
    err=err/N
    return err[0]

def solveclose(X_train, y_train):#function of Closed Form
    w=np.random.uniform(-1,1,X_train.shape[1])#weight
    w=np.dot(np.linalg.inv((np.dot(X_train.transpose(),X_train))),np.dot(X_train.transpose(),y_train))
    
    err=errMSE(X_train, y_train,w)
    #err=err**0.5        
    return w,err

def solvequery(X_train, y_train,X_test,y_test,t):#use close form to solve locally weighted linear regression
    X_query=X_test[:,0]#query points
    err=0;
    y_predict=[]
    for i in range(0,len(X_query)):
        x=X_query[i]*np.ones(X_train.shape[0])#query point
        r=[]
        R=[]
        r=np.exp(-(X_train[:,0]-x)**2/(2*t*t))
        R=r*np.eye(X_train.shape[0])
        w=np.random.uniform(-1,1,X_train.shape[1])#weight
        w=np.dot(np.linalg.inv((np.dot(np.dot(X_train.transpose(),R),X_train))),np.dot(np.dot(X_train.transpose(),R),y_train))
        err=err+(np.dot(X_test[i],w)-y_test[i])**2
        y_predict.append(np.dot(X_test[i],w)[0])
    N=X_test.shape[0]
    err=err/N
        #err=errMSE(X_test, y_test,w)
    #err=err**0.5        
    return w,err,y_predict
    
# For this problem, we use data generator instead of real dataset
def data_generator(size,noise_scale=0.05):
    xs = np.random.uniform(low=0,high=3,size=size)

    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + np.sin(3*xs) + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys

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

def plottest(X_test,y_test,y_test_predict):#make plot
    plt.scatter(X_test[:,0],y_test_predict,label='predict')
    plt.scatter(X_test[:,0],y_test,label='real')
    plt.legend()
    plt.xlabel('X_test')
    plt.ylabel('predict result of y_test')
            
def main():
    noise_scales = [0.05,0.2]

    # for example, choose the first kind of noise scale
    noise_scale = noise_scales[0]

    # generate the data form generator given noise scale
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)

    X_train, X_test=normalizaion(X_train,X_test)
    w,errtrain = solveclose(X_train, y_train)
    errtest=errMSE(X_test, y_test,w)
    # bandwidth parameters
    sigma_paras = [0.1,0.2,0.4,0.8,1.6]
    fig1 = plt.figure()
    print('err: ', errtest)
    y_test_predict=[]#predict result of test set
    y_test_predict=np.dot(X_test,w)
    plottest(X_test,y_test,y_test_predict)
    
    fig2 = plt.figure()
    t=0.2
    wq,errtestquery,y_predict = solvequery(X_train, y_train,X_test,y_test,t)
    #errtestquery=errMSE(X_test, y_test,wq)#test error of locally weighted LR
    print('errquery: ', errtestquery)
    #print('ypre: ', y_predict)
    plottest(X_test,y_test,y_predict)
    
    fig3 = plt.figure()
    t=2
    wqq,errtestquery,y_predict = solvequery(X_train, y_train,X_test,y_test,t)
    #errtestquery=errMSE(X_test, y_test,wqq)#test error of locally weighted LR
    print('errquery: ', errtestquery)
    plottest(X_test,y_test,y_predict)
    
main()




