import tensorflow as tf
import numpy as np
import hdf5storage
import scipy.io as sio
from numpy import cov

class Reg_Binary_0_1(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
    def __call__(self, x):      
        x1 = tf.sigmoid(500*x)
        x2 = tf.sigmoid(500*x)
        ##
        Yg = None
        for i in range(0,self.bands):
            if Yg is not None:
              Yg = Yg + tf.roll(x2, shift=i, axis=1)
              #Cv = CV + cov(x2,tf.roll(x2, shift=i, axis=1))
            else:
              Yg = x2
              #Cv = 0
        Dz = tf.math.reduce_mean(Yg)+tf.math.reduce_std(Yg)
        ##
        Dx = tf.reduce_sum(tf.multiply(tf.square(x1),tf.square(1-x1)))
        Dx = tf.math.reduce_mean(Dx)
        regularization = self.parameter*(Dz + Dx) 
        return regularization

class CassiLayer(tf.keras.layers.Layer):
    def __init__(self, bands, Factor, Mm, Nn, **kwargs):
        self.bands = bands
        self.Factor   = Factor
        self.M = Mm
        self.N = Nn
        self.myregularizer=Reg_Binary_0_1(self.Factor)
        super(CassiLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bands': self.bands})
        return config

    def build(self, input_shape):
        #       
        H_init = np.random.normal(0, 0.01, size=(self.M, self.N)).astype(np.float32) 
        sio.savemat('Init.mat',{'Init':H_init})    
        H_init = tf.constant_initializer(H_init)      
        self.dmd = self.add_weight(name="CAC", initializer=H_init, shape=(self.M, self.N),regularizer=self.myregularizer, trainable=True) #regularizer = algo
        super(CassiLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mask = self.dmd
        mask = tf.sigmoid(500*self.dmd)        
        mask = tf.expand_dims(mask,axis=0)
        mask = tf.expand_dims(mask,axis=-1)
        return mask
