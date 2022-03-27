import tensorflow as tf
import numpy as np
from SensingProduct_200x200 import *
from Main_cassi_layer_200x200 import CassiLayer
from SensingProduct_200x200 import SensingDirect
from SensingProduct_200x200 import SensingTranspose
from SensingProduct_200x200 import ProductLb
from SensingProduct_200x200 import ProductLb2


def Gen2(size=None, training=True, weigh_decay=0.0, output_bands=100, bands=1, upsampling = 1, Factor = 10**-6):
    if size is None:
        size = [160, 184, 25]
    end_points = {}

    reg = tf.keras.regularizers.l2(weigh_decay)
    L = 25
    Mm = 512
    Nn = 512
    Xx     = tf.keras.layers.Input(size, name='Entrada1') # 256,256,100
    ShearF = tf.keras.layers.Input([3], name='Entrada2') # 256,256,100
    Mask = CassiLayer(bands=bands, Factor=Factor,Mm=Mm,Nn=Nn, name='CassiLayer')(Xx) # 256,256,1   
    #### Sensing
    G = SensingDirect(bands=bands, Mm=Mm,Nn=Nn, name='DirectPr_Init')(Xx,Mask,ShearF)
    #####################
    #
    #
    Gf = tf.keras.layers.Conv2D(L, 3, strides=1, padding="SAME", kernel_regularizer=reg, name='en_Gf_1')(G)
    Gf = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gf)
    #
    Gf = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Gf_2')(Gf)
    Gf = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gf)    
    #
    Gf = tf.keras.layers.Conv2D(1, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Gf_3')(Gf)
    Gf = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gf)    
    #
    Gf = tf.keras.layers.Conv2D(1, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Gf_4')(Gf)
    Gf = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gf)                
    ##    
    #
    netM = tf.keras.layers.Flatten()(Gf)
    net3 = tf.keras.layers.Dense(64, activation='sigmoid')(netM)
    net3 = tf.keras.layers.Dense(32, activation='sigmoid')(net3)
    net3 = tf.keras.layers.Dense(16, activation='sigmoid')(net3)
    ShS = tf.keras.layers.Dense(3, activation='sigmoid',name='OtAp')(net3)        
    #####################
    Gt = SensingTranspose(bands=bands, Mm=Mm,Nn=Nn, name='TransPr_Init')(G,Mask,ShS)
    #####################
    #
    Gg = tf.keras.layers.Conv2D(L, 3, strides=1, padding="SAME", kernel_regularizer=reg, name='en_Gt_1')(Gt)
    Gg = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gg)
    #
    Gg = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Gt_2')(Gg)
    Gg = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gg)    
    #
    Gg = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Gt_3')(Gg)
    Gg = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gg)    
    #
    Gg = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Gt_4')(Gg)
    Gg = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Gg)                
    ##
    #####################
    ## Upper-Arm
    #####################
    Gt = SensingDirect(bands=bands, Mm=Mm,Nn=Nn, name='DirectPr_2')(Gg,Mask,ShS)
    #
    F1 = InverseProduct(bands=bands, Mm=Mm,Nn=Nn, name='InversePr_1')(Gt,Mask,ShS)
    F1 = SensingTranspose(bands=bands, Mm=Mm,Nn=Nn, name='TransPr_2')(F1,Mask,ShS)
    #
    F1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding="SAME", kernel_regularizer=reg, name='en_Up_1')(F1)
    F1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(F1)
    #
    F1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Up_2')(F1)
    F1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(F1)    
    #
    F1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Up_3')(F1)
    F1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(F1)    
    #
    F1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_Up_4')(F1)
    F1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(F1)            
    #
    #####################
    ## Bottom-Arm
    #####################    
    Gt = SensingTranspose(bands=bands, Mm=Mm,Nn=Nn, name='TransPr_3')(G,Mask,ShS)  
    F1 = ProductLb2(bands=bands, Mm=Mm,Nn=Nn, name='Pw_1')(Gg) - F1 
    #
    Z1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding="SAME", kernel_regularizer=reg, name='en_1_1')(F1)
    Z1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Z1)
    Z1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_1_2')(Z1)
    Z1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Z1)
    Z1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_1_3')(Z1)
    Z1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Z1)    
    Z1 = tf.keras.layers.Conv2D(L, 3, strides=1, padding='SAME', kernel_regularizer=reg, name='en_1_4')(Z1)
    Z1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(Z1)        

    f = L
    layers = []
    x = Z1
    for i in range(0, 3):
      x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
      x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
      layers.append(x)
      x = tf.keras.layers.MaxPooling2D() (x)
      f = f*2
    ff2 = f 
    
    #bottleneck 
    j = len(layers) - 1
    x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
    x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
    x = tf.keras.layers.Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = tf.concat([x, layers[j]],axis=-1)
    j = j -1 
    ##
    #upsampling 
    for i in range(0, 2):
      ff2 = ff2//2
      f = f // 2 
      x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
      x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
      x = tf.keras.layers.Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
      x = tf.concat([x, layers[j]],axis=-1)
      j = j -1     
    ##  
    x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
    x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same') (x)
    outputs = tf.keras.layers.Conv2D(L, 1, activation='sigmoid') (x)   
    #      
    Gg = SensingDirect(bands=bands, Mm=Mm,Nn=Nn, name='DirectPr_3')(outputs,Mask,ShS)
    LssM = tf.reduce_mean(tf.norm(G - Gg, ord=1))/(Mm*Nn)
    return tf.keras.Model([Xx,ShearF], [outputs,ShS,LssM], name="Generator_lambda2")
