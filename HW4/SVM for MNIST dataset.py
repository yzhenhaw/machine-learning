import numpy as np

from sklearn import datasets, svm
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

mnist = fetch_mldata('MNIST original', data_home='./')

#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target

N = len(images)
np.random.seed(1234)
inds = np.random.permutation(N)
images = np.array([images[i] for i in inds])
targets = np.array([targets[i] for i in inds])

# Normalize data
X_data = images/255.0
Y = targets

# Train/test split
X_train, y_train = X_data[:10000], Y[:10000]
X_test, y_test = X_data[-10000:], Y[-10000:]

clf= svm.SVC(C=1,gamma=1)
clf.fit(X_train, y_train)

y_predict=clf.predict(X_test)
yy=y_predict-y_test# difference between y_predict and y_test
error=np.count_nonzero(yy)/10000

print("accuracy=",1-error)



kfold = KFold(n_splits=5)

Cset=[1,3,5]
gammaset=[0.05,0.1,0.5,1.0]

for C in Cset:
    for gamma in gammaset:
        error1=0
        for train_index,test_index in kfold.split(X_train):
            X_train1, X_test1 = X_train[train_index], X_train[test_index]
            y_train1, y_test1 = y_train[train_index], y_train[test_index]

            #train classifier for the subset of train data
            clf = svm.SVC(C=C,gamma=gamma)
            clf.fit(X_train1, y_train1)
            y_predict1=clf.predict(X_test1)
            yy1=y_predict1-y_test1
            error1+=np.count_nonzero(yy1)/len(y_test1)/5
        print("C=",C,"gamma=",gamma,"accuracy=",1-error1)