import tensorflow as tf
import numpy as np
import hdf5storage
import scipy.io as sio
##############################################
## SensingDirect
##############################################
class SensingDirect(tf.keras.layers.Layer):
    def __init__(self, bands, Mm, Nn, **kwargs):
        self.bands = bands
        self.M = Mm
        self.N = Nn        
        super(SensingDirect, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bands': self.bands})
        return config

    def call(self, inputs, mask, ShearF, **kwargs):
      Aux1 = tf.multiply(mask, inputs)
      Aux1 = tf.pad(Aux1, [[0, 0], [0, 0], [0, self.bands - 1], [0, 0]],name="padsensing")
      Yg = None
      Lt = np.linspace(0, self.bands-1, self.bands) 
      for i in range(0,self.bands):
          if Yg is not None:
            Cx = Lt[i] + ShearF[0,0]*Lt[i] - ShearF[0,1]*Lt[i]**2 + ShearF[0,2]*Lt[i]**3
            Cx = int(Cx)
            if Cx>=self.bands:
              Cx = self.bands-1
            elif Cx<0:
              Cx = 0
            Yg = Yg + tf.roll(Aux1[:,:,:,i], shift=Cx, axis=2)
          else:
            Yg = Aux1[:,:,:,i] 
      Yg = tf.expand_dims(Yg,axis=-1)
      Yg  = tf.math.divide(Yg,self.bands)
      return Yg
              
##############################################
## SensingTranspose
##############################################
class SensingTranspose(tf.keras.layers.Layer):
    def __init__(self, bands, Mm, Nn, **kwargs):
        self.bands = bands
        self.M = Mm
        self.N = Nn        
        super(SensingTranspose, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bands': self.bands})
        return config

    def call(self, inputs, mask, ShearF, **kwargs):
      X = None
      Mk = mask[0,:,:,0]
      Yg    = inputs[:,:,:,0]
      Lt = np.linspace(0, self.bands-1, self.bands)
      for i in range(0,self.bands):
          if X is not None:
            Cx = Lt[i] + ShearF[0,0]*Lt[i] - ShearF[0,1]*Lt[i]**2 + ShearF[0,2]*Lt[i]**3
            Cx = int(Cx)
            if Cx>=self.bands:
              Cx = self.bands-1
            elif Cx<0:
              Cx = 0            
            Ab = tf.roll(Yg, shift=-Cx, axis=2)  
            Ax = tf.expand_dims(tf.multiply(Mk, Ab[:,:,0:self.N]), -1)# - tf.expand_dims(tf.multiply(mask1, Yg1[:, :, i:self.M+i]), -1)
            X = tf.concat([X, Ax], axis=-1)    
          else:
            Ab = tf.roll(Yg, shift=0, axis=2)
            X = tf.expand_dims(tf.multiply(Mk,Ab[:,:,0:self.N]), -1)# - 
      X = self.bands*X
      return X     
##############################################
## InverseProduct
##############################################
class InverseProduct(tf.keras.layers.Layer):
    def __init__(self, bands,Mm, Nn, **kwargs):
        self.bands = bands
        self.M = Mm
        self.N = Nn        
        super(InverseProduct, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bands': self.bands})
        return config
    
    def build(self, input_shape):
        Lambda = tf.constant_initializer(1)
        Tau = tf.constant_initializer(1)
        Psi = np.zeros([self.M,self.N+self.bands-1])
        Psi = tf.constant_initializer(Psi)
        self.Lambda = self.add_weight(name="Lbd", initializer=Lambda, shape=(1),trainable=True)
        self.Tau = self.add_weight(name="Tau", initializer=Tau, shape=(1),trainable=True,constraint=tf.keras.constraints.MaxNorm(max_value=1, axis=0))
        self.Psi = self.add_weight(name="Psi", initializer=Psi, shape=(self.M,self.N+self.bands-1),trainable=True)
        super(InverseProduct, self).build(input_shape)    

    def call(self, inputs, mask,ShearF, **kwargs):
      Yg = inputs[:,:,:,0]
      Mk1 = tf.pad(mask[:,:,:,0], [[0, 0], [0, 0], [0, self.bands - 1]])
      Mk1 = tf.multiply(Mk1,Mk1)
      Mk = None
      Lt = np.linspace(0, self.bands-1, self.bands)
      #
      for i in range(0,self.bands):
          if Mk is not None:
              Cx = Lt[i] + ShearF[0,0]*Lt[i] - ShearF[0,1]*Lt[i]**2 + ShearF[0,2]*Lt[i]**3
              Cx = int(Cx)
              if Cx>=self.bands:
                Cx = self.bands-1
              elif Cx<0:
                Cx = 0              
              Mk = Mk + tf.roll(Mk1, shift=Cx, axis=2)
          else:
                Mk = Mk1                
      #
      X = None
      Mk = tf.math.divide(Mk,self.bands)
      Mk = Mk/(self.Lambda) + tf.ones(Mk.shape)
      Mk = tf.math.reciprocal(Mk, name=None)
      Yg1 = tf.multiply((self.Tau**2)*Mk+(1-self.Tau**2)*self.Psi,Yg)
      
      for i in range(0,self.bands):
          if X is not None:
            Cx = Lt[i] + ShearF[0,0]*Lt[i] - ShearF[0,1]*Lt[i]**2 + ShearF[0,2]*Lt[i]**3
            Cx = int(Cx)
            if Cx>=self.bands:
              Cx = self.bands-1
            elif Cx<0:
              Cx = 0              
            Ab1 = tf.roll(Yg1, shift=-Cx, axis=2)  
            Ax = tf.expand_dims(tf.multiply(mask[0,:,:,0], Ab1[:,:,0:self.N]), -1)
            X = tf.concat([X, Ax], axis=-1)    
          else:
            Ab1 = tf.roll(Yg1, shift=0, axis=2)
            X = tf.expand_dims(tf.multiply(mask[0,:,:,0],Ab1[:,:,0:self.N]), -1)   
      X = self.bands*X
      X = X/(self.Lambda**2)
      #X = tf.math.divide(X,tf.math.reduce_max(X))
             
      return X     
##############################################
## Multiplication
##############################################
class ProductLb(tf.keras.layers.Layer):
    def __init__(self, bands, Mm, Nn, **kwargs):
        self.bands = bands
        self.M = Mm
        self.N = Nn        
        super(ProductLb, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bands': self.bands})
        return config

    def build(self, input_shape):
        Nu1 = tf.constant_initializer(1)
        self.Nu1 = self.add_weight(name="Nu", initializer=Nu1, shape=(1),trainable=True)
        super(ProductLb, self).build(input_shape) 

    def call(self, X, **kwargs):
        Nu= self.Nu1
        X = X*Nu
        return X    
##############################################
## Multiplication
##############################################
class ProductLb2(tf.keras.layers.Layer):
    def __init__(self, bands, Mm, Nn, **kwargs):
        self.bands = bands
        self.M = Mm
        self.N = Nn        
        super(ProductLb2, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bands': self.bands})
        return config

    def build(self, input_shape):
        Nu1 = tf.constant_initializer(1)
        self.Nu1 = self.add_weight(name="Nu", initializer=Nu1, shape=(1),trainable=True)
        super(ProductLb2, self).build(input_shape) 

    def call(self, X, **kwargs):
        Nu= self.Nu1
        X = X/(Nu**2)
        return X            