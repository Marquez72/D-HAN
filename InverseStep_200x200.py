import tensorflow as tf
import numpy as np
import hdf5storage
import scipy.io as sio


class InverseStep(tf.keras.layers.Layer):
    def __init__(self, bands,Mm, Nn, **kwargs):
        self.bands = bands
        self.M = Mm
        self.N = Nn        
        super(InverseStep, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bands': self.bands})
        return config

#    def build(self, input_shape):
#        Mu_init = tf.constant_initializer(1)
#        self.Mu = self.add_weight(name="Mu", initializer=Mu_init, shape=(1), trainable=True) #regularizer = algo
#        super(InverseStep, self).build(input_shape)


    def call(self, inputs, mask, **kwargs):
      Yg = inputs[:,:,:,0]
      mask1 = mask[0,:,:,0]
      Mk1 = tf.pad(mask[:,:,:,0], [[0, 0], [0, 0], [0, self.bands - 1]])
      Mk1 = tf.multiply(Mk1,Mk1)
      Mk = None
      for i in range(0,self.bands):
          if Mk is not None:
                Mk = Mk + tf.roll(Mk1, shift=i, axis=2)
          else:
                Mk = Mk1                
      X = None
      Mk = Mk + tf.ones(Mk.shape)
      Mk = tf.math.reciprocal(Mk, name=None)
      Yg1 = tf.multiply(Mk,Yg)
      for i in range(0,self.bands):
          Ab = tf.roll(Yg, shift=-i, axis=2)
          Ab1 = tf.roll(Yg1, shift=-i, axis=2)
          if X is not None:
                Ax = tf.expand_dims(tf.multiply(mask1, Ab[:,:,0:self.N]), -1)# - tf.expand_dims(tf.multiply(mask1, Yg1[:, :, i:self.M+i]), -1)
                X = tf.concat([X, Ax], axis=-1)    
                Ax = tf.expand_dims(tf.multiply(mask1, Ab1[:,:,0:self.N]), -1)
                X1 = tf.concat([X1, Ax], axis=-1)    
          else:
                X = tf.expand_dims(tf.multiply(mask1,Ab[:,:,0:self.N]), -1)# - 
                X1 = tf.expand_dims(tf.multiply(mask1,Ab1[:,:,0:self.N]), -1)   
      X = tf.math.divide(X,tf.math.reduce_max(X))
      X1 = tf.math.divide(X1,tf.math.reduce_max(X1))
             
      return X, X1