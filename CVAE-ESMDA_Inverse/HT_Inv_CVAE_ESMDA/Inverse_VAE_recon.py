import numpy as np
import os
os.system('module load cuda/9.0')
os.system('module load cudnn/9.0_v7.6.4')
import keras
from keras.models import Model, Sequential,model_from_json
from keras import regularizers
from keras import backend as K
from scipy.io import savemat, loadmat
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.layers import Reshape, Lambda, Input, Dense, Flatten, Conv2D, Conv2DTranspose, Dropout
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical
K.set_image_data_format('channels_last') 
import warnings
warnings.filterwarnings("ignore")
import forward_model
from ES_MDA import ES_MDA

# initialization
nx=80
ny=40
ncores=20 # for parallel computation
Num_ens=500
Dim_latent=400
Na=4
Alpha=np.array([9.333,7.0,4.0,2.0])

obs=np.loadtxt('obs_error.txt')
var_obs = np.ones_like(obs)
obs=np.array([obs])
obs=obs.T

# set the error for different kinds of measurements
var_obs[:3792]=1e-4 # HHT
#var_obs[3792:3812]=1e-6 # c
#var_obs[3812:]=0.01 # ERT
R=np.diag(var_obs)

s=np.zeros((Num_ens,Dim_latent,Na+1))
s[:,:,0]=np.random.randn(Num_ens,Dim_latent)


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

json_file = open('decoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)
# load weights into new model
decoder.load_weights("decoder.h5")
# load mean_std K for reconstruction
data1 = loadmat('mean_std.mat')
x_mean = data1['x_mean']
x_std = data1['x_std']
KS_final_ens = decoder.predict(s[:,:,len(Alpha)])  # KS images for each ens
KS_final_ens[:, :, :, :1] = KS_final_ens[:, :, :, :1] * x_std[:, :, :] + x_mean[:, :, :]
s_final=np.mean(s[:,:,:],axis=0)    # KS_mean for each step
s_final=s_final.T
KS_mean = decoder.predict(s_final)  # final KS images for the mean s
KS_mean[:, :, :, :1] = KS_mean[:, :, :, :1] * x_std[:, :, :] + x_mean[:, :, :]
savemat('results_esmda.mat',{'s':s,'KS_final_ens':KS_final_ens,'KS_mean':KS_mean}) 
