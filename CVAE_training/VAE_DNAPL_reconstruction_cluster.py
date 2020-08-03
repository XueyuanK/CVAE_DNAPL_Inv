import numpy as np
import os
os.system('module load cuda/9.0')
os.system('module load cudnn/9.0_v7.6.4')
import copy
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, Sequential
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
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import Callback
import hdf5storage

# total number of epochs
n_epochs = 200 

class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight 

    def on_epoch_end (self, epoch, logs={}):
        def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
            L = np.ones(n_epoch)
            period = n_epoch/n_cycle
            step = (stop-start)/(period*ratio) # linear schedule
            for c in range(n_cycle):
                v , i = start , 0
                while v <= stop and (int(i+c*period) < n_epoch):
                    L[int(i+c*period)] = v
                    v += step
                    i += 1
            return L
        #new_weight= frange_cycle_linear(0.0,1.0,n_epochs,4)
        new_weight=1.0
        #K.set_value(self.weight, new_weight[epoch])
        K.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.weight)))

def sampling(input_param):
    """
    sampling the latent space from a Gaussian distribution:
    # Input
        input_param: mean and log of variance of q(z|x)
    # Output
        z: sampled latent space vector
    """
    #mean and log(var):
    z_mean, z_log_var = input_param
    #dimensions:
    dim_1 = K.shape(z_mean)[0]
    dim_2 = K.int_shape(z_mean)[1]
    #sampling:
    norm_sample = K.random_normal(shape=(dim_1, dim_2))
    return z_mean + K.exp(0.5 * z_log_var) * norm_sample


#encoder network:

#regularization coefficient:
l_encode=0.0

#input:
DNAPL_input = Input(shape=(40,80, 4), name = 'DNAPL')

#CNN/pooling layer 1:
x = Conv2D(16, (3, 3), strides=2, padding='same',kernel_regularizer = regularizers.l2(l_encode))(DNAPL_input)
x = BatchNormalization(axis = 3)(x)
x = Activation('relu')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)

#CNN/pooling layer 2:
x = Conv2D(32, (3, 3), strides=2, padding='same',kernel_regularizer = regularizers.l2(l_encode))(x)
x = BatchNormalization(axis = 3)(x)
x = Activation('relu')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)

#CNN/pooling layer 3:
x = Conv2D(64, (3, 3), strides=1, padding='same',kernel_regularizer = regularizers.l2(l_encode))(x)
x = BatchNormalization(axis = 3)(x)
x = Activation('relu')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)

#flattening:
shape = K.int_shape(x)
x = Flatten()(x)

#fully connected layer 1:
x = Dense(1600,kernel_regularizer = regularizers.l2(l_encode))(x)
x = BatchNormalization(axis = 1)(x)
x = Activation('relu')(x)


#output:
z_mean = Dense(400,activation='linear',name='z_mean')(x)
z_log_var = Dense(400,activation='linear',name='z_log_var')(x)
z = Lambda(sampling, output_shape=(400,), name='latent_encode')([z_mean, z_log_var])

#set encoder model:
encoder = Model(DNAPL_input, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

#decoder network:
#regularization coefficient
l_decode=0.0

latent_input = Input(shape=(400,), name='latent_decode') #input

#fully connected layer 1:
x = Dense(1600, kernel_regularizer = regularizers.l2(l_decode))(latent_input)
x = BatchNormalization(axis = 1)(x)
x = Activation('relu')(x)

#fully connected layer 2:
x = Dense((shape[1])*(shape[2])*shape[3], kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 1)(x)
x = Activation('relu')(x)

#reshaping:
x = Reshape((shape[1], shape[2], shape[3]))(x)

#dCNN layer 1:
x = Conv2DTranspose(64, (3, 3), strides=1, padding='same', kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 3)(x)
x = Activation('relu')(x)
#x = UpSampling2D((2, 2))(x)

#dCNN layer 2:
x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 3)(x)
x = Activation('relu')(x)
#x = UpSampling2D((2, 2))(x)

#dCNN layer 3:
x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 3)(x)
x = Activation('relu')(x)
#x = UpSampling2D((2, 2))(x)

#output:
outputs1 = Conv2DTranspose(1, (3, 3), activation='linear', padding='same', name='K')(x)
outputs2 = Conv2DTranspose(3, (3, 3), activation='softmax', padding='same', name='S')(x)
outputs = keras.layers.concatenate([outputs1, outputs2])

#set decoder model:
decoder = Model(latent_input, outputs, name='decoder')
decoder.summary()

#set VAE model
vae_outputs = decoder(encoder(DNAPL_input)[2])
vae = Model(DNAPL_input, vae_outputs, name='vae')

# the starting value of weight is 0
# define it as a keras backend variable
weight = K.variable(1.)
# wrap the loss as a function of weight
def vae_loss(weight):
    def loss (y_true, y_pred):
        l_kl=1.0 
        l_mse_s=1.0
        
        y_true_m=K.flatten(y_true[:,:,:,0])
        y_pred_m=K.flatten(y_pred[:,:,:,0])
        mse_loss = 3200*mse(y_true_m, y_pred_m)#mse loss
        
        #y_true_s=y_true[:,:,:,1]+2*y_true[:,:,:,2]+3*y_true[:,:,:,3]       
        #y_pred_s=y_pred[:,:,:,1]+2*y_pred[:,:,:,2]+3*y_pred[:,:,:,3]
        #mse_loss_s = 3200*mse(K.flatten(y_true_s), K.flatten(y_pred_s))#mse loss for s
        
        y_true_c=K.flatten(y_true[:,:,:,1:])
        y_pred_c=K.flatten(y_pred[:,:,:,1:])
        cross_entropy_loss = 80*40*binary_crossentropy(y_true_c, y_pred_c)#cross_entropy
        
        kl_loss = - 0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  #KL loss:
        #vae_loss1 = mse_loss+l_mse_s*mse_loss_s+cross_entropy_loss +weight*K.mean(kl_loss)    #total loss
        vae_loss1 = mse_loss+cross_entropy_loss +weight*K.mean(kl_loss)    #total loss
        return vae_loss1
    return loss

def acc_pred(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true[:,:,:,1:], axis=-1), K.argmax(y_pred[:,:,:,1:], axis=-1)), K.floatx())

def k_mse(y_true, y_pred):
    y_true_m=K.flatten(y_true[:,:,:,0])
    y_pred_m=K.flatten(y_pred[:,:,:,0])
    return mse(y_true_m, y_pred_m)

#def scon_mse(y_true, y_pred):
#    y_true_s=y_true[:,:,:,1]+2*y_true[:,:,:,2]+3*y_true[:,:,:,3]      
#    y_pred_s=y_pred[:,:,:,1]+2*y_pred[:,:,:,2]+3*y_pred[:,:,:,3]
#    return mse(K.flatten(y_true_s), K.flatten(y_pred_s))


#compile VAE:
Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
vae.compile(optimizer=Adam, loss=vae_loss(weight), metrics=[acc_pred,k_mse])
vae.summary()

#load/read data
data = hdf5storage.loadmat('KS_cla4_604_cha_rel_poi.mat')
innapl = data['KS'] #input images and output images are the same
outnapl = data['KS'] 

#check data shape:
print(innapl.shape)
print(outnapl.shape)

#put KS in right shape:
innapl_dummy = innapl.reshape(4,40,80,-1)
innapl_new1 = np.zeros((innapl_dummy.shape[3],innapl_dummy.shape[0],innapl_dummy.shape[1],innapl_dummy.shape[2]))
innapl_new = np.zeros((innapl_dummy.shape[3],innapl_dummy.shape[1],innapl_dummy.shape[2],innapl_dummy.shape[0]))
for i in range(innapl_dummy.shape[3]):
    innapl_new1[i,:,:,:] = innapl_dummy[:,:,:,i]
for i in range(innapl_dummy.shape[0]):
    innapl_new[:,:,:,i] = innapl_new1[:,i,:,:]
del innapl_dummy
del innapl_new1

#check innapl_new shape:
print(innapl_new.shape)


outnapl_dummy = outnapl.reshape(4,40,80,-1)
outnapl_new1 = np.zeros((outnapl_dummy.shape[3],outnapl_dummy.shape[0],outnapl_dummy.shape[1],outnapl_dummy.shape[2]))
outnapl_new = np.zeros((outnapl_dummy.shape[3],outnapl_dummy.shape[1],outnapl_dummy.shape[2],outnapl_dummy.shape[0]))
for i in range(outnapl_dummy.shape[3]):
    outnapl_new1[i,:,:,:] = outnapl_dummy[:,:,:,i]
for i in range(outnapl_dummy.shape[0]):
    outnapl_new[:,:,:,i] = outnapl_new1[:,i,:,:]
del outnapl_dummy
del outnapl_new1

#check outnapl_new shape:
print(outnapl_new.shape)

#add channel dimension:
y_train = outnapl_new 
x_train = innapl_new  
del innapl_new
del outnapl_new

#check shapes:
print(x_train.shape)
print(y_train.shape)

#normalize input/outputs:
x_train_norm = copy.deepcopy(x_train) # need to normalize the K field
y_train_norm = copy.deepcopy(y_train)

#find normalizing parameters:
x_mean = np.mean(x_train[:,:,:,:1], axis=0)
y_mean = np.mean(y_train[:,:,:,:1], axis=0)
x_std = np.std(x_train[:,:,:,:1], axis=0)
y_std = np.std(y_train[:,:,:,:1], axis=0)

#check shapes:
print(x_mean.shape)
print(y_mean.shape)
print(x_std.shape)
print(y_std.shape)

#normalize input/outputs:
x_train_norm[:,:,:,:1] = (x_train[:,:,:,:1]-x_mean)/x_std
y_train_norm[:,:,:,:1] = (y_train[:,:,:,:1]-y_mean)/y_std

#check shapes:
print(x_train_norm.shape)
print(y_train_norm.shape)

#training:
N=60000 # number of data to be used
val_split=0.1 #validation split
#vae.load_weights('vae_rec.h5') #to load saved weights, if necessary
history=vae.fit(x=x_train_norm[:N,:,:,:],y=y_train_norm[:N,:,:,:],
        epochs=n_epochs,batch_size=32,shuffle=True,validation_split=val_split,callbacks=[AnnealingCallback(weight)])
vae.save_weights('vae_rec.h5') #to save optimized weights

#calculate VAE predictions (normalized):
z_mean_pred, z_log_pred, z_pred= encoder.predict(x_train_norm) #encoder output
y_decoded_pred = decoder.predict(z_pred) #decoder output

#check shape:
print(y_decoded_pred.shape)

#calculate VAE predictions:
y_pred=y_decoded_pred
y_pred[:,:,:,:1]=y_decoded_pred[:,:,:,:1]*y_std[:,:,:]+y_mean[:,:,:] #predicted output

#check shape:
print(y_pred.shape)

#accuracy of train/test/validation:
print('train err for Sn', np.sum(np.equal(np.argmax(y_train[:int(N-val_split*N),:,:,1:], axis=-1),
                np.argmax(y_pred[:int(N-val_split*N),:,:,1:], axis=-1))+0)/(3200*int(N-val_split*N)))
print('validation err for Sn', np.sum(np.equal(np.argmax(y_train[int(N-val_split*N):N,:,:,1:], axis=-1),
                np.argmax(y_pred[int(N-val_split*N):N,:,:,1:], axis=- 1))+0)/(3200*int(val_split*N)))
print('test err for Sn', np.sum(np.equal(np.argmax(y_train[N:,:,:,1:], axis=-1),
                np.argmax(y_pred[N:,:,:,1:], axis=-1))+0)/(3200*int(innapl.shape[2]-N)))

print('train rmse for K:',np.sqrt(np.mean((y_pred[:int(N-val_split*N),:,:,0]-y_train[:int(N-val_split*N),:,:,0])**2)))
print('validation rmse for K:',np.sqrt(np.mean((y_pred[int(N-val_split*N):N,:,:,0]-y_train[int(N-val_split*N):N,:,:,0])**2)))
print('test rmse for K:',np.sqrt(np.mean((y_pred[N:,:,:,0]-y_train[N:,:,:,0])**2)))


x_train1=x_train[60000:60100,:,:,:] #save a small portion of testing set to plot the results; 
y_pred1=y_pred[60000:60100,:,:,:]
z_mean_pred=z_mean_pred[60000:60100,:]
z_log_pred=z_log_pred[60000:60100,:]
z_pred=z_pred[60000:60100,:]
savemat('reconstructed_realization_testing.mat',{'x_train1':x_train1,'y_pred1':y_pred1,'z_mean_pred':z_mean_pred,'z_log_pred':z_log_pred,'z_pred':z_pred})

#plot some results from the training sets
k=500
y_ref=np.argmax(y_train[k,:,:,1:],axis=-1)
y_est=np.argmax(y_pred[k,:,:,1:],axis=-1)
minv = y_ref.min()
maxv = y_ref.max()
#minv = 0
#maxv = 1
xx = np.linspace(0.5, 40, 80)
yy = np.linspace(0.5, 20, 40)
XX, YY = np.meshgrid(xx, yy)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
im = axes[0].pcolormesh(XX,YY,y_ref, vmin=minv, vmax=maxv, cmap=plt.get_cmap('jet'))
axes[0].set_title('(a) True', loc='left')
axes[0].set_aspect('equal')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].axis([XX.min(), XX.max(), YY.min(), YY.max()])
axes[1].pcolormesh(XX, YY, y_est, vmin=minv, vmax=maxv, cmap=plt.get_cmap('jet'))
axes[1].set_title('(b) Estimate', loc='left')
axes[1].set_xlabel('x (m)')
axes[1].set_aspect('equal')
axes[1].axis([XX.min(), XX.max(), YY.min(), YY.max()])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('rec_ref_training.png', dpi=fig.dpi)
plt.close(fig)

#generate some random samples (unconditional realizations)
z_pred=np.random.randn(300,400)
#predict
y_decoded_pred = decoder.predict(z_pred) #decoder output
#check shape:
print(y_decoded_pred.shape)
#calculate VAE predictions:
y_pred=y_decoded_pred
y_pred[:,:,:,:1]=y_decoded_pred[:,:,:,:1]*y_std[:,:,:]+y_mean[:,:,:] #predicted output
savemat('unconditional_sample_300.mat',{'y_pred':y_pred})
