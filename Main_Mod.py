from Main_discriminator import inference
import tensorflow as tf
from Unrolling2 import Gen2
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
################################################################################################
##
################################################################################################

L = 25
M = int(512)
N = int(512)
Sf = 1
batch_size = 4
Bx  = tf.linspace(0.0, 0.2, 5)
Bx = np.float32(Bx)
Cx = np.ones(10)
Cx[0] = 0
Cx = np.float32(Cx)
#
data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),layers.experimental.preprocessing.RandomRotation(0.2),])
     
def normalize_0_to_1(mat):
    for i in range(mat.shape[0]):
        mat[i]=(mat[i]-mat[i].min())/(mat[i].max()-mat[i].min())
    return mat

def shuffle_crop(batch_size):
    index = np.random.choice(range(3790-251),batch_size)+251  
    Rx=np.random.choice(range(5),1)
    Rx1=np.random.choice(range(5),1)
    new_data=[]
    ShearF = []
    Lind  = np.random.choice(range(100-L),batch_size)
    Af = [0.8+Bx[Rx],0.072,0.0013]
    Af = np.float32(Af)
    for i in range(batch_size):
        path = './Data_Temporal/%d.mat' % index[i]
        img = hdf5storage.loadmat(path)['data']     
        #
        img = np.float32(img)
        img = img[:,:,Lind[i]:Lind[i]+L] 
        #
        for j in range(0,L):
          Ax = tf.expand_dims(cv2.resize(img[:,:,j], (512, 512)),axis=-1) 
          if j==0:
            Im = Ax
          else:
            Im = tf.concat([Im,Ax], axis=-1)
        img = tf.expand_dims(Im,axis=0)
        #        
        img = data_augmentation(img)
        img = tf.squeeze(img,axis=0)
        img = np.float32(img)
        #
        if tf.math.reduce_max(img)==0:
            print(i)

        img = np.float16(np.divide(img,tf.math.reduce_max(img)))  
        new_data.append(img)
        ShearF.append(Af)
    new_data=np.array(new_data)
    ShearF=np.array(ShearF)
    return new_data, ShearF

################################################################################################
##
################################################################################################
#
gen = Gen2(size=[M, N, L], training=True, weigh_decay=1e-8, output_bands=L, bands=L, upsampling = 1, Factor = 1e8)
generator_optimizer = tf.keras.optimizers.Adam(1e-45)
#

@tf.function
def train_step(images,ShearF):
    ## This method returns a helper function to compute cross entropy loss
    def Gen_loss(weight):
        def lossimage_2(y_true, y_pred):
            # spatial_loss =  tf.reduce_mean(1-tf.image.ssim(y_pred,y_true,1))
            spatial_loss = tf.reduce_mean(tf.norm(y_pred - y_true, ord=1))/(M*N*L)+tf.reduce_mean(
                1 - tf.image.ssim_multiscale(y_pred, y_true, 1))
            val = weight*(spatial_loss)
            return val
        return lossimage_2
    ##    
    def Loss_Rf(weight):
        def lossimage_2(y_ground_truth, y_output):
            Ax = tf.cast(y_ground_truth,tf.float32)
            Bx = tf.cast(y_output,tf.float32)        
            val = weight*tf.cast(tf.reduce_mean(tf.norm(Ax - Bx, ord=1)),tf.float32)
            return val
        return lossimage_2         
    ##    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      [Gen_Unet,ShS,LssM] = gen([images, ShearF],training=True)
      #
      Gen_lossU = Gen_loss(1)(images, Gen_Unet)
      Gen_lossSh = Loss_Rf(100)(ShearF, ShS)
      Sum_G = (Gen_lossU+Gen_lossSh+LssM)# + GenL
      #          

    gradients_of_generator = gen_tape.gradient(Sum_G, gen.trainable_variables)  
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
  
    return Sum_G#,disc_loss

################################################################################################
##
################################################################################################

def run_gan():

    #gen.load_weights("checkpoints/genw_CoherenceTemporal.tf")

    epochs = 10000
    Loss = 1e6
    iter = 50  
    Mx = 0  
    for i in range(epochs):
        LsG_Ep = 0
        LsD_Ep = 0
        for j in range(iter):
            # train discriminator       
            training_data_batch = []
            training_data_batch, ShearF = shuffle_crop(batch_size=batch_size)  # (bs, 256, 256, 100)          
            training_data_batch = tf.convert_to_tensor(training_data_batch,dtype=tf.float32)
            Sum_G = train_step(training_data_batch,ShearF)
            LsG_Ep = LsG_Ep + Sum_G
            #LsD_Ep = disc_loss + LsD_Ep
        Zm =Recon(gen=gen)    
        #if ((LsG_Ep/iter)<Loss):
        if Zm>Mx:
            Mx = Zm
            Loss = (LsG_Ep/iter)
            CA = gen.get_weights()[0]
            sio.savemat('Opt_50.mat',{'Iter':CA})   
            CA = tf.sigmoid(500*CA)          
            CA = np.float32(255*CA)
            cv2.imwrite('CA.jpg',CA)
            print("epoch=", i, " Loss_Gen=", LsG_Ep/iter, " Mean_PSNR=", Mx)
            gen.save_weights('checkpoints/genw_CoherenceTemporal.tf')                    
    return 0


if __name__ == '__main__':
    run_gan()  
