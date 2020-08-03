import numpy as np
from scipy import array, linalg, dot
import math

def ES_MDA(num_ens,m_ens,Z,prod_ens,alpha,CD,corr,numsave=2):
    # Z is the obs; prob_ens is the sim_obs; CD is the error
    # note that in this func, the shape of the array is (Num_nodes,Num_ens)
    varn=1-1/math.pow(10,numsave)
    # Initial Variavel 
    # Forecast step
    yf = m_ens.T                      # Non linear forward model, transform to (Num_nodes,Num_ens)
    df = prod_ens.T                   # Observation Model, transform to (Num_nodes,Num_ens)
    ym = np.array(yf.mean(axis=1))    # Mean of the y_f
    dm = np.array(df.mean(axis=1))    # Mean of the d_f
    ym=ym.reshape(ym.shape[0],1)    
    dm=dm.reshape(dm.shape[0],1)    
    dmf = yf - ym
    ddf = df - dm
    
    Cmd_f = (np.dot(dmf,ddf.T))/(num_ens-1);  # The cros-covariance matrix
    Cdd_f = (np.dot(ddf,ddf.T))/(num_ens-1);  # The auto covariance of predicted data
    
    # Perturb the vector of observations
    R = linalg.cholesky(CD,lower=True) #Matriz triangular inferior
    U = R.T   #Matriz R transposta
    p , w =np.linalg.eig(CD)
    
    aux = np.repeat(Z,num_ens,axis=1)
    mean = 0*(Z.T)

    noise=np.random.multivariate_normal(mean[0], np.eye(len(Z)), num_ens).T
    d_obs = aux+math.sqrt(alpha)*np.dot(U,noise)  
    
    # Analysis step
    u, s, vh = linalg.svd(Cdd_f+alpha*CD); v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ss = np.diag(diagonal**(-1))
    K=np.dot(Cmd_f,(np.dot(np.dot(v,ss),(u.T))))
    # Use Kalman covariance
    if len(corr)>0:
        K = corr*K
        
    ya = yf + (np.dot(K,(d_obs-df)))
    m_ens = ya
    return m_ens.T # tranform back to (Num_ens,Num_nodes)

# reference
#@article{emerick2013ensemble,title={Ensemble smoother with multiple data assimilation},author={Emerick, Alexandre A and Reynolds, Albert C},
#  journal={Computers \& Geosciences},
#  volume={55},
#  pages={3--15},
#  year={2013},
#  publisher={Elsevier}}
# https://www.sciencedirect.com/science/article/pii/S0098300412000994