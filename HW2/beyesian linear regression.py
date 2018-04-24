import numpy as np
import matplotlib.pyplot as plt

pi=3.1415926
# generate heat plot (image-like) for gaussian
def plot_heat(mean,sig):
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

# we defined a class for sequential bayesian learner
class bayesian_linear_regression(object):

    # initialized with covariance matrix(sigma), mean vector(mu) and prior(beta)
    def __init__(self,sigma,mu,beta):
        self.sigma = sigma
        self.mu = mu
        self.beta = beta
        self.count=0#count the number of iteration

    # you need to implement the update function
    # when received additional design matrix phi and continuous label t
    def update(self,phi,t):
        sigmanew=np.linalg.inv(np.linalg.inv(self.sigma)+self.beta*np.dot(phi.transpose(),phi))
        munew=np.dot(sigmanew,(self.beta*np.dot(phi.transpose(),t)+np.dot(np.linalg.inv(self.sigma),self.mu)))
        self.sigma=sigmanew
        self.mu=munew
        self.count=self.count+1
        if(self.count==1):
            fig2 = plt.figure()
            plot_heat(self.mu,self.sigma)
            print("sigma:",self.sigma)
            print("mean:",self.mu)
        elif(self.count==10):
            fig3 = plt.figure()
            plot_heat(self.mu,self.sigma)
            print("sigma:",self.sigma)
            print("mean:",self.mu)
        elif(self.count==20):
            fig4 = plt.figure()
            plot_heat(self.mu,self.sigma)
            print("sigma:",self.sigma)
            print("mean:",self.mu)
        #return(self.sigma,self.mu)

def data_generator(size,scale):
    x = np.random.uniform(low=-3, high=3, size=size)
    rand = np.random.normal(0, scale=scale, size=size)
    y = 0.5 * x - 0.3 + rand
    phi = np.array([[x[i], 1] for i in range(x.shape[0])])
    t = y
    return phi, t


def main():
    # initialization
    alpha = 2
    sigma_0 = np.diag(1.0/alpha*np.ones([2]))
    mu_0 = np.zeros([2])
    beta = 1.0
    blr_learner = bayesian_linear_regression(sigma_0, mu_0, beta=beta)
    
    fig1 = plt.figure()
    plot_heat(mu_0,sigma_0)
    print("sigma:",sigma_0)
    print("mean:",mu_0)
    
    num_episodes = 20

    for epi in range(num_episodes):
        phi, t = data_generator(1,1.0/beta)
        blr_learner.update(phi,t)
        

main()


