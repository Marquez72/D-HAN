import tensorflow as tf
import numpy as np
import hdf5storage
import scipy.io as sio
import cv2
import math


def Recon(gen=None):
    L = 25
    Bd = np.linspace(3862, 3871, num=10)#[2279,2280,2281,2282,2283,2284,2285,2286,2289,2290]
    Zz = np.zeros([10])
    Bx  = [0,0.05,0.1,0.15,0.2,0,0.05,0.1,0.15,0.2,0,0.05,0.1,0.15,0.2,0,0.05,0.1,0.15,0.2]#tf.linspace(0.0, 0.2, 5)
    Bx = np.float32(Bx)      
    
    for Nm in range(0,10):
        Af = [0.8+Bx[Nm],0.072,0.0013]
        #Af = [1,0.072,0.0013]
        Af = np.float32(Af)
        ShearF = []
        ShearF.append(Af)
        ShearF=np.array(ShearF)
        Ph = '/content/gdrive/My Drive/Doctorado/ArticulosMiguel/Datasets/Data_Temporal/%d.mat' %Bd[Nm]
        #Ph = '/content/gdrive/My Drive/Canada/Spectral/testing_data/%d.mat' %Bd[Nm]
        
        testing_data=hdf5storage.loadmat(Ph)['data'] #(10,256,256,24)
        testing_data = testing_data[:,:,0:L]
        #
        for j in range(0,L-1):
          Ax = tf.expand_dims(cv2.resize(testing_data[:,:,j], (512, 512)),axis=-1) 
          if j==0:
            Im = Ax
          else:
            Im = tf.concat([Im,Ax], axis=-1)
        Im = tf.concat([Im,Ax], axis=-1)    
        testing_data = tf.expand_dims(Im,axis=0)
        #          
        testing_data = np.float16(np.divide(testing_data,tf.math.reduce_max(testing_data))) 
        testing_data = np.array(testing_data)
        g12 = []
        [g12,Sh,_] = gen.predict([testing_data,ShearF])
        psnr = 0.0
        for w in range(L):
            single_mse=np.mean((testing_data[0,:,:,w] - g12[0,:,:,w]) ** 2)
            psnr+=20 * math.log10(1 / math.sqrt(single_mse))        
        Zz[Nm] = psnr/L
        Tf = None
        ContC = 0
        for Ci in range(0,4):
            Tf1 = None
            for Cj in range(0,6):
                if Tf1 is not None:
                    Tf1 = np.concatenate([Tf1,255*g12[0,:,:,ContC]],1)
                else:
                    Tf1 = 255*g12[0,:,:,ContC]
                ContC = ContC + 1
            if Tf is not None:
              Tf = np.concatenate([Tf,Tf1],0)
            else:
              Tf = Tf1
        #
        Array = np.array(Sh)
        file = open("Results/Shearing.txt", "a+")
        content = str(Array)
        file.write(content + '\n')
        file.close()                 
        #
        X = np.float32(g12[0,:,:,:])     
        X = np.float32(255*np.divide(X,tf.math.reduce_max(X)))  
        cv2.imwrite('Results/Re_%d.jpg'%Nm,Tf)
        cv2.imwrite('Results/RRG_%d.jpg'%Nm,X[:,:,[21,11,4]])    
        #sio.savemat('Results/Re_%d.mat'%Nm,{'X':X}) 
    Zm = np.mean(Zz)
    sio.savemat('Results/Psnr.mat',{'Zz':Zz}) 
    Array = np.array(Zz/L)
    file = open("Results/PSNR.txt", "a+")
    content = str(Array)
    file.write(content + '\n')
    file.close()      
    return Zm