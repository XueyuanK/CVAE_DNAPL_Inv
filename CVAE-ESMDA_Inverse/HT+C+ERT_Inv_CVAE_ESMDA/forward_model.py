import numpy as np
import os
os.system('module load cuda/9.0')
os.system('module load cudnn/9.0_v7.6.4')
from keras.models import Model, Sequential, model_from_json
from keras import regularizers
from keras import backend as K
from scipy.io import savemat, loadmat
from keras.losses import mse, binary_crossentropy
from keras.layers import Reshape, Lambda, Input, Dense, Flatten, Conv2D, Conv2DTranspose
from keras.layers import Activation, ZeroPadding2D, BatchNormalization
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical
K.set_image_data_format('channels_last') 
import warnings
warnings.filterwarnings("ignore")


class Model:
    def __init__(self, params=None):

        self.ncores = params['ncores']

        if params is not None:
            self.nx = params['nx']
            self.ny = params['ny']
        else:
            raise ValueError("You have to provide relevant parameters")

    def run_model(self, s):
        '''run forward in comsol
        '''
        # load json and create model
        ############## need to optimize the code here, move the decoder as a parameter read by the func (global var)
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
        KS = decoder.predict(s)  # initial KS images
        KS[:, :, :, :1] = KS[:, :, :, :1] * x_std[:, :, :] + x_mean[:, :, :]

        HK=KS[:,:,:,0]
        Sn = np.argmax(KS[:, :, :, 1:], axis=-1)
        Sn = Sn*0.2 # class 0 is sn=0; class 1 is sn=0.2;class 2 is sn=0.4

        # transform all the ensembles of s to input files of COMSOL
        ngrid = self.nx * self.ny
        coor_k = np.zeros([ngrid, 3])
        coor_sn = np.zeros([ngrid, 3])
        ngroup=int(s.shape[0]/self.ncores)
        os.chdir('.//forward_model_parallel')

        simul_obs=[]

        for igroup in range(0,ngroup,1):
            HK_group=HK[igroup*self.ncores:(igroup+1)*self.ncores,:,:]
            Sn_group=Sn[igroup*self.ncores:(igroup+1)*self.ncores,:,:]
            for ireaz in range(0, self.ncores, 1):
                str_dir = './/parallel' + str(ireaz + 1)
                os.chdir(str_dir)
                index = 0
                for j in range(0, self.ny, 1):
                    for i in range(0, self.nx, 1):
                        coor_k[index, 0] = i*0.5
                        coor_k[index, 1] = j*0.5
                        coor_k[index, 2] = HK_group[ireaz,j,i]

                        coor_sn[index, 0] = i*0.5
                        coor_sn[index, 1] = j*0.5
                        coor_sn[index, 2] = Sn_group[ireaz,j,i]

                        index = index + 1
                np.savetxt("inputK.txt", coor_k, fmt="%f  %f  %f")
                np.savetxt("inputSn.txt", coor_sn, fmt="%f  %f  %f")
                os.chdir("../")
            # run HT/SP-COMSOL
            while (not os.path.exists('output_obs.txt')) :
                os.system('./run_test.sh')

            # read the observation
            simul_obs_group = np.loadtxt('output_obs.txt')  # collect all the obs (for one gruop) in this file
            os.remove('output_obs.txt')
            if simul_obs == []:
                simul_obs = simul_obs_group
            else:
                simul_obs=np.hstack((simul_obs,simul_obs_group))
        os.chdir("../")
        return simul_obs.T