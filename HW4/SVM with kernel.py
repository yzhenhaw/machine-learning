import numpy as np 
from sklearn import svm

np.random.seed(3)

mean_1 = [ 2.0 , 0.2 ]
cov_1 = [ [ 1 , .5 ] , [ .5 , 2.0 ]]

mean_2 = [ 0.4 , -2.0 ]
cov_2 = [ [ 1.25 , -0.2 ] , [ -0.2, 1.75 ] ]

x_1 , y_1 = np.random.multivariate_normal( mean_1 , cov_1, 15).T
x_2 , y_2 = np.random.multivariate_normal( mean_2 , cov_2, 15).T

import matplotlib.pyplot as plt 

plt.plot( x_1 , y_1 , 'x' )
plt.plot( x_2 , y_2 , 'ro')
plt.axis('equal')
plt.show()

X = np.zeros((30,2))
X[0:15,0] = x_1
X[0:15,1] = y_1
X[15:,0] = x_2
X[15:,1] = y_2

y = np.zeros(30)
y[0:15] = np.ones(15)
y[15:] = -1 * np.ones(15)

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
#C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=1.0),
          svm.SVC(kernel='linear', C=100.0),
          svm.SVC(kernel='rbf', C=1),
          svm.SVC(kernel='rbf', C=3))
models = (clf.fit(X, y) for clf in models)
#SVnumber =(clf.n_support_[0] for clf in models)
#
#print("number of Support Vector in these four cases",SVnumber)



# title for the plots
titles = ('SVC with linear kernel C=1.0',
          'SVC with linear kernel C=100.0',
         'SVC with RBF kernel C=1.0',
         'SVC with RBF kernel C=3.0')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)


xx, yy = make_meshgrid(X[:, 0], X[:, 1])

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(-4, 6)
    ax.set_ylim(-6, 3)
#    ax.set_xlabel('Sepal length')
#    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    print("Number of support vector", clf.n_support_)
plt.show()
#for clf in models:
#    print(clf.n_support_)