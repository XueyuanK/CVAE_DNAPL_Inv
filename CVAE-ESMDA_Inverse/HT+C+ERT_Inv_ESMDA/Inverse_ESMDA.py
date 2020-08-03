import numpy as np
import warnings
warnings.filterwarnings("ignore")
import forward_model
from ES_MDA import ES_MDA
from scipy.io import savemat, loadmat

# initialization
nx=80
ny=40
ns=nx*ny*2 # both K and S need to be identified
ncores=20 # for parallel computation
Num_ens=500
Na=4
Alpha=np.array([9.333,7.0,4.0,2.0])

obs=np.loadtxt('obs_conti_error.txt')
var_obs = np.ones_like(obs)
obs=np.array([obs])
obs=obs.T

# set the error for different kinds of measurements
var_obs[:3792]=1e-4 # HHT
var_obs[3792:3812]=1e-6 # c
var_obs[3812:]=1.0 # ERT
R=np.diag(var_obs)

s=np.zeros((Num_ens,ns,Na+1))
data1 = loadmat('KS_1000_for_trad_ESMDA.mat')
KS_ini = data1['KS']
KS_ini = KS_ini.T
KS_ini = KS_ini[:Num_ens,:]
s[:,:,0]=KS_ini

forward_params = params = {'nx': nx, 'ny': ny,'ncores': ncores}
model = forward_model.Model(forward_params)

for t in range(len(Alpha)):
    sim_obs=model.run_model(s[:,:,t]) # shape of sim_obs (Num_ens,Num_obs)
    print('RMSE ite_', t, ' : ', np.sqrt(np.mean((np.mean(sim_obs,axis=0)-obs.flatten())**2))) # not the exact RMSE definition
    s[:,:,t+1] = ES_MDA(Num_ens, s[:,:,t], obs, sim_obs, Alpha[t], R, [], 2)
    s_tem=s[:,:,t+1]
    savemat('./s_tem' + str(t+1) + '.mat', {'s_tem':s_tem}) # save s for each step
sim_obs=model.run_model(s[:,:,len(Alpha)]) # shape of sim_obs (Num_ens,Num_obs)
print('RMSE ite_', len(Alpha), ' : ', np.sqrt(np.mean((np.mean(sim_obs,axis=0)-obs.flatten())**2)))

savemat('results_esmda.mat',{'s':s}) 
