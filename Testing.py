from Main_discriminator import inference
import tensorflow as tf
from Unrolling import generator
import numpy as np
import h5py
from GenerarRt import Recon
import hdf5storage
from scipy import ndimage
import scipy.io as sio
from google.colab.patches import cv2_imshow
import cv2
from scipy import ndimage
from tensorflow.keras import layers
import math
################################################################################################
##
################################################################################################

L = 25
M = int(256)
N = int(256)
Sf = 1
batch_size = 1
Bx  = tf.linspace(0.0, 0.2, 5)
Bx = np.float32(Bx)
Cx = np.ones(6)
Cx[0] = 0
Cx = np.float32(Cx)
#
################################################################################################
##
################################################################################################

gen = generator(size=[M, N, L], training=True, weigh_decay=1e-8, output_bands=L, bands=L, upsampling = 1, Factor = 1e8)
dis = inference(size=[M, N, L], training=True, weigh_decay=1e-8)
#
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

gen.load_weights("checkpoints/genw_CoherenceSpectral.tf")

#Bd = [451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470]
Bd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
Zz = np.zeros([450])
#Bx  = [0,0.05,0.1,0.15,0.2,0,0.05,0.1,0.15,0.2,0,0.05,0.1,0.15,0.2,0,0.05,0.1,0.15,0.2]#tf.linspace(0.0, 0.2, 5)
Bx = 0.2 #p.float32(Bx)      

for Nm in range(0,36):
    Af = [0.8+Bx,0.072,0.0013]
    #Af = [0.0,0.0,0.0]
    Af = np.float32(Af)
    ShearF = []
    ShearF.append(Af)
    ShearF=np.array(ShearF)
    #Ph = '/content/gdrive/My Drive/Doctorado/ArticulosMiguel/Canada/Spectral/testing_data/%d.mat' %Bd[Nm]
    Ph = '/content/gdrive/My Drive/Canada/Spectral/training_data/M%d.mat' %int(Nm+1)
    
    testing_data=hdf5storage.loadmat(Ph)['data'] #(10,256,256,24)
    #testing_data = testing_data[128:128+256,128:128+256,(np.floor(np.linspace(0, testing_data.shape[2]-1, num=L))).astype(np.int)]
    testing_data = testing_data[0:256,0:256,:]

    testing_data = np.float16(np.divide(testing_data,tf.math.reduce_max(testing_data))) 
    testing_data = tf.expand_dims(testing_data,axis=0)
    testing_data = np.array(testing_data)
    g12 = []
    [g12,Sh,_] = gen.predict([testing_data,ShearF])
    psnr = 0.0
    for w in range(L):
        single_mse=np.mean((testing_data[0,:,:,w] - g12[0,:,:,w]) ** 2)
        psnr+=20 * math.log10(1 / math.sqrt(single_mse))        
    psnr = psnr/L
    Zz[Nm] = psnr
    X = np.float32(g12[0,:,:,:])     
    X = np.float32(255*np.divide(X,tf.math.reduce_max(X)))     
    sio.savemat('Re_%d.mat'%Bd[Nm],{'X':X}) 
    #
    Array = np.array(Sh)
    file = open("Sh_Pr_WLinear.txt", "a+")
    content = str(Array)
    file.write(content + '\n')
    file.close()                 
    #
    sio.savemat('Psnr_Pr_WLinear.mat',{'Zz':Zz}) 
