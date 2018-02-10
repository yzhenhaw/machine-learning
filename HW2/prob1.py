import numpy as np
import matplotlib.pyplot as plt


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
    for i in range(0,len(X_train)):
        X_train[i]=np.append(X_train[i],1)
    for i in range(0,len(X_test)):   
        X_test[i]=np.append(X_test[i],1)
    return X_train, X_test
        
def main():
    noise_scales = [0.05,0.2]

    # for example, choose the first kind of noise scale
    noise_scale = noise_scales[0]

    # generate the data form generator given noise scale
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)

    #X_train, X_test=normalizaion(X_train,X_test)
    
    # bandwidth parameters
    sigma_paras = [0.1,0.2,0.4,0.8,1.6]

main()




