import numpy as np
from scipy.io import savemat, loadmat
import warnings
import os
import math
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
        HK=s[:,:3200]
        Sn=s[:,3200:]

        # transform all the ensembles of s to input files of COMSOL
        ngrid = self.nx * self.ny
        coor_k = np.zeros([ngrid, 3])
        coor_sn = np.zeros([ngrid, 3])
        ngroup=int(s.shape[0]/self.ncores)
        os.chdir('.//forward_model_parallel')

        simul_obs=[]

        for igroup in range(0,ngroup,1):
            HK_group=HK[igroup*self.ncores:(igroup+1)*self.ncores,:]
            Sn_group=Sn[igroup*self.ncores:(igroup+1)*self.ncores,:]
            for ireaz in range(0, self.ncores, 1):
                str_dir = './/parallel' + str(ireaz + 1)
                os.chdir(str_dir)
                index = 0
                for j in range(0, self.ny, 1):
                    for i in range(0, self.nx, 1):
                        coor_k[index, 0] = i*0.5
                        coor_k[index, 1] = j*0.5
                        coor_k[index, 2] = HK_group[ireaz,index]

                        coor_sn[index, 0] = i*0.5
                        coor_sn[index, 1] = j*0.5
                        coor_sn[index, 2] = 1.0/ (math.exp(-1*Sn_group[ireaz,index])+1)

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