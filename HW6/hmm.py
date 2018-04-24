import numpy as np
np.random.seed(17)

observation1=np.array([2,2,4,1,2,1,2,3,2,1])
observation2=np.array([2,4,1,2,3,2,1,1,4])
def forback(pi,A,phi,K,obs):
    alpha1=np.zeros((len(obs),K))
    beta1=np.zeros((len(obs),K))
    alpha1[0,:]=pi*phi[:,1]
    for i in range(1,len(obs)):
        alpha1[i,:]=[sum(alpha1[i-1,j]*A[j,k] for j in range(K))*phi[k,obs[i]-1] for k in range(K)]
    beta1[len(obs)-1,:]=np.ones(K)
    for i in range(len(obs)-1):
        x=len(obs)-2-i
        beta1[x,:]=[sum([beta1[x+1,j]*A[k,j]*phi[j,obs[x+1]-1] for j in range(K)]) for k in range(K)]
    return alpha1,beta1

def getgammaxi(alpha,beta,A,phi,obs):
    gamma=(alpha*beta).T/np.sum(alpha*beta,1)
    xi=np.zeros((K,K,alpha.shape[0]-1))
    for i in range(K):
        for j in range(K):
            for k in range(alpha.shape[0]-1):
                xi[i,j,k]=alpha[k,i]*A[i,j]*beta[k+1,j]*phi[j,obs[k+1]-1]
    xi=xi/np.sum(np.sum(xi,0),0)
    return gamma,xi

def BaumWelch(pi,A,phi,K):
    Anew=np.zeros((A.shape[0],A.shape[1]))
    phinew=np.zeros((phi.shape[0],phi.shape[1]))
    for epoch in range(1):
        alpha1,beta1=forback(pi,A,phi,K,observation1)
        alpha2,beta2=forback(pi,A,phi,K,observation2)
        gamma1,xi1=getgammaxi(alpha1,beta1,A,phi,observation1)
        gamma2,xi2=getgammaxi(alpha2,beta2,A,phi,observation2)
        #pi=(gamma1[:,0]+gamma2[:,0])/2
        
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                Anew[i,j]=(np.sum(xi1,2)+np.sum(xi2,2))[i,j]/(np.sum(gamma1[:,:-1],1)+np.sum(gamma2[:,:-1],1))[i]
        for i in range(phi.shape[0]):
            for k in range(phi.shape[1]):
                obs1=np.zeros(len(observation1))
                obs2=np.zeros(len(observation2))
                for j in range(len(observation1)):
                    if (observation1[j]-1)==k:
                        obs1[j]=1
                    else:
                        obs1[j]=0
                for j in range(len(observation2)):
                    if (observation2[j]-1)==k:
                        obs2[j]=1
                    else:
                        obs2[j]=0    
                phinew[i,k]=(np.sum(gamma1*obs1,1)+np.sum(gamma2*obs2,1))[i]/(np.sum(gamma1,1)+np.sum(gamma2,1))[i]
        
        A=Anew
        phi=phinew
    return A,phi

def Viterbi(A,phi):
    state=np.zeros(4)
    obser=np.zeros(4)
    obser[0]=0
    for i in range(1,4):
        state[i]=np.argmax(phi[:,int(obser[i-1])])
        state[i]=np.argmax(A[int(state[i]),:])    
        obser[i]=np.argmax(phi[int(state[i]),:])
    DNA=['A','C','G','T']
    output=[]
    for i in range(4):
        output.append(DNA[int(obser[i])])
    return output

# for K = 2 , you should use the following parameters to 
K=2
# transition matrix
Initial_A_2 = np.array([
    [0.4,0.6],
    [0.6,0.4]
])

# emission matrix
Initial_phi_2 = np.array([
    [0.5, 0.1, 0.2, 0.2],
    [0.1, 0.5, 0.1, 0.3]
])
    
pi2=np.ones(K)*1/K
A2,phi2=BaumWelch(pi2,Initial_A_2,Initial_phi_2,2)    
output2=Viterbi(A2,phi2)
print("transition of 2 states\n",A2)
print("emission of 2 states\n",phi2)
print("most likely sequence of 2 states\n",output2)

# for K = 4 , you should use the following parameters to initialize
# transition matrix
Initial_A_4 = np.array([
    [0.3, 0.1, 0.2, 0.4],
    [0.1, 0.2, 0.4, 0.3],
    [0.2, 0.4, 0.3, 0.1],
    [0.4, 0.3, 0.1, 0.2]]
)

# emission matrix
Initial_phi_4 = np.array([
    [0.5, 0.1, 0.2, 0.2],
    [0.1, 0.5, 0.1, 0.3],
    [0.1, 0.2, 0.5, 0.2],
    [0.3, 0.1, 0.1, 0.5]
])
K=4
pi4=np.ones(K)*1/K
A4,phi4=BaumWelch(pi4,Initial_A_4,Initial_phi_4,4)  
output4=Viterbi(A4,phi4)  
print("transition of 4 states\n",A4)
print("emission of 4 states\n",phi4)
print("most likely sequence of 4 states\n",output4)


    





