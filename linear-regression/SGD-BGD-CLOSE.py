import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def solveSGD(X_train, y_train):#function of SGD
    maxep=500 #max epochs
    rate=0.0003#learning rate0.072
    ep=0 #epoch
    error=[] #learning error of each epoch
    w=np.random.uniform(-1,1,14)#weight
    done=True#flag to end the loop
    while done:
        err=0#MSE
        order=np.arange(len(y_train))
        np.random.shuffle(order)
        for k in range (0,len(y_train)):
            i=order[k]
            w=w-rate*(np.dot(w,X_train[i])-y_train[i])*X_train[i]
            err=err+(np.dot(w,X_train[i])-y_train[i])**2
        N=X_train.shape[0]
        err=err/N
        error.append(err)           
        ep=ep+1
        if(ep>=maxep):
            break
        if(err<0.0001):
            break
    return w,error

def solveBGD(X_train, y_train):#function of Batch Gradient Descent
    maxep=500 #max epochs
    rate=0.0005#learning rate
    ep=0 #epoch
    error=[] #learning error of each epoch
    w=np.random.uniform(-1,1,14)#weight
    done=True#flag to end the loop
    while done:
        err=0#MSE
        gradient=0
        for i in range (0,len(y_train)):
            gradient=gradient+(np.dot(w,X_train[i])-y_train[i])*X_train[i]#calculate the gradient
        w=w-rate*gradient#update weight
        for i in range (0,len(y_train)):
            err=err+(np.dot(w,X_train[i])-y_train[i])**2
        N=X_train.shape[0]
        err=err/N
        error.append(err)           
        ep=ep+1
        if(ep>=maxep):
            break
        if(err<0.0001):
            break
    return w,error

def solveclose(X_train, y_train):#function of Closed Form
    err=0
    error=[] #learning error of each epoch
    w=np.random.uniform(-1,1,14)#weight
    w=np.dot(np.linalg.inv((np.dot(X_train.transpose(),X_train))),np.dot(X_train.transpose(),y_train))
        #w=w-rate*gradient
    for i in range (0,len(y_train)):
        err=err+(np.dot(w,X_train[i])-y_train[i])**2
    N=X_train.shape[0]
    err=err/N
    error.append(err)           
    return w,error

def test(w,X_test,y_test):#calculate the testing error
    err=0#MSE
    for i in range (0,len(y_test)):
        err=err+(np.dot(w,X_test[i])-y_test[i])**2
    N=X_test.shape[0]
    err=err/N
    return err
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

    # Preprocess your data - Normalization, adding a constant feature
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
    
    #w,error = solveSGD(X_train, y_train)
    #w,error = solveBGD(X_train, y_train)
    w,error = solveclose(X_train, y_train)
    
    
    
    errtest = test (w,X_test,y_test)
    plt.plot(error)
    plt.ylabel('MSE')
    plt.xlabel('number of epochs')
    plt.show()
    
    np.set_printoptions(threshold=20)#print all the data
    print ('w: ', w)
    print ('testerror: ', errtest)
    print ('trainerror: ', error[-1])

    # Collect train and test errors
    # Use your implementation of the mse function
    train_errs.append(error[-1])
    test_errs.append(errtest)

    print('Mean training error: ', np.mean(train_errs))
    print('Mean test error: ', np.mean(test_errs))

    
