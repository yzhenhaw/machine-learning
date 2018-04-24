import numpy as np
import matplotlib.pyplot as plt

pi=3.1415926
# feel free to read the two examples below, try to understand them
# in this problem, we require you to generate contour plots

# generate contour plot for function z = x^2 + 2*y^2
def plot_contour():

    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            xmu=np.array([X[j][i]-mean[0],Y[j][i]-mean[1]])
            Z[j][i] = (1/((2*pi)**2*np.linalg.det(sig))**0.5)*np.exp(-0.5*np.dot(np.dot(xmu.transpose(),np.linalg.inv(sig)),xmu))
            

    plt.clf()
    plt.axis("square")
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)
    axes = plt.gca()
    axes.set_xlim([-3,3])
    axes.set_ylim([-3,3])
    plt.show()


# generate heat plot (image-like) for gaussian
def plot_heat():
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            xmu=np.array([X[j][i]-mean[0],Y[j][i]-mean[1]])
            Z[j][i] = (1/((2*pi)**2*np.linalg.det(sig))**0.5)*np.exp(-0.5*np.dot(np.dot(xmu.transpose(),np.linalg.inv(sig)),xmu))

    plt.clf()
    img = Z
    plt.imshow(img, interpolation='none', extent=[-3.0, 3.0, -3.0, 3.0],cmap="plasma")
    plt.colorbar()
    plt.axis("square")
    plt.show()

def data_generator(size,mean,noise_scale):
    xs = np.random.uniform(low=-3,high=3,size=size)
    ys = xs + mean + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys

# This function receives the parameters of a multivariate Gaussian distribution
# over variables x_1, x_2 .... x_n as input and compute the marginal
#
def marginal_for_guassian(sigma,mu,given_indices):
    # given selected indices, compute marginal distribution for them
    mean=mu[given_indices]#mean of each given variable of Gussian
    si=sigma[given_indices,given_indices]#sigma of each given variable of Gussian   
    size=(100,1)
    X,Y=data_generator(size,mean,si)
    return X,Y


def conditional_for_gaussian(sigma,mu,given_indices,given_values):
    # given some indices that have fixed value, compute the conditional distribution
    # for rest indices
    # P(xb|xa), given indices is for a
    meanxa=mu[given_indices]
    meanxb=np.delete(mu,given_indices)
    xn=range(0,sigma.shape[0])#indice of whole array
    indiceb=np.delete(xn,given_indices)#indice left for b
    sigaa=sigma[np.ix_(given_indices,given_indices)]
    sigbb=sigma[np.ix_(indiceb,indiceb)]
    sigab=sigma[np.ix_(given_indices,indiceb)]
    sigba=sigma[np.ix_(indiceb,given_indices)]
    sig=sigbb-np.dot(np.dot(sigba,np.linalg.inv(sigaa)),sigab)#covariance of conditional Gaussian distribution
    mean=meanxb+np.dot(np.dot(sigba,np.linalg.inv(sigaa)),given_values-meanxa)#mean of conditional Gaussian distribution
    #D=len(mean)#dimension of conditional Gaussian
    #X = np.random.uniform(low=-3,high=3,size=(100,D))#variables in conditional Gaussian
    #Z=(1/((2*pi)**D*np.linalg.det(sig))**0.5)*np.exp(-0.5*np.dot(np.dot((X-mean).transpose(),np.linalg.inv(sig))),(X-mean))
    return sig,mean


test_sigma_1 = np.array(
    [[1.0, 0.5],
     [0.5, 1.0]]
)

test_mu_1 = np.array(
    [0.0, 0.0]
)

test_sigma_2 = np.array(
    [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.0, 1.5],
     [0.0, 0.0, 2.0, 0.0],
     [0.0, 1.5, 0.0, 4.0]]
)

test_mu_2 = np.array(
    [0.5, 0.0, -0.5, 0.0]
)

indices_1 = np.array([0])

indices_2 = np.array([1,2])
values_2 = np.array([0.1,-0.2])

X,Y=marginal_for_guassian(test_sigma_1, test_mu_1, indices_1)
plt.scatter(X,Y)
plt.xlabel('X1')
plt.ylabel('Y')

sig,mean=conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)
print(sig,mean)
#x1 = np.random.uniform(low=-3,high=3,size=(100,1))
#x4 = np.random.uniform(low=-3,high=3,size=(100,1))

plot_contour()
#plot_heat()

#marginal_for_guassian(test_sigma_1, test_mu_1, indices_1)
#conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)