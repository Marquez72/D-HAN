
import tensorflow as tf
import numpy as np
import hdf5storage
import scipy.io as sio




def HIDDSP(pretrained_weights=None, input_size=(512, 512, 31), depth=64, bands=31):

  inputs = Input(shape=input_size,name='image')
  X0, y,H = DD_CASSI_Layer(output_dim=input_size, input_dim=input_size,parm1=1e-9)(inputs)
  #y = Forward_D_CASSI(inputs,H)
  #X0 = Transpose_D_CASSI(y,H)
  #--------Stage 1---------------------
  # - h step--
  conv_r1 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X0)
  conv_r1 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r1)
  conv_r1 = Add()([X0,conv_r1])
  conv_r1 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r1)
  # - x step --
  # H^T(Hf-y)
  X1 = Lambda(GradientCASSI)([X0,y,H])
  X1_prior = Subtract()([X0,conv_r1])
  X1_prior = Mu_parameter()(X1_prior)

  X1 = Add()([X1_prior,X1])
  X1 = Lambda_parameter()(X1)
  X1 = Subtract(name='X1')([X0,X1])

    #--------Stage 2---------------------
  # - h step--
  conv_r2 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X1)
  conv_r2 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r2)
  conv_r2 = Add()([X1,conv_r2])
  conv_r2 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r2)
  # - x step --
  # H^T(Hf-y)
  X2 = Lambda(GradientCASSI)([X1,y,H])
  X2_prior = Subtract()([X1,conv_r2])
  X2_prior = Mu_parameter()(X2_prior)

  X2 = Add()([X2_prior,X2])
  X2 = Lambda_parameter()(X2)
  X2 = Subtract(name='X2')([X1,X2])

    #--------Stage 3---------------------
  # - h step--
  conv_r3 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X2)
  conv_r3 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r3)
  conv_r3 = Add()([X2,conv_r3])
  conv_r3 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r3)
  # - x step --
  # H^T(Hf-y)
  X3 = Lambda(GradientCASSI)([X2,y,H])
  X3_prior = Subtract()([X2,conv_r3])
  X3_prior = Mu_parameter()(X3_prior)

  X3 = Add()([X3_prior,X3])
  X3 = Lambda_parameter()(X3)
  X3 = Subtract(name='X3')([X2,X3])

      #--------Stage 4---------------------
  # - h step--
  conv_r4 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X3)
  conv_r4 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)
  conv_r4 = Add()([X3,conv_r4])
  conv_r4 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)
  # - x step --
  # H^T(Hf-y)
  X4 = Lambda(GradientCASSI)([X3,y,H])
  X4_prior = Subtract()([X3,conv_r4])
  X4_prior = Mu_parameter()(X4_prior)

  X4 = Add()([X4_prior,X4])
  X4 = Lambda_parameter()(X4)
  X4 = Subtract(name='X4')([X3,X4])

        #--------Stage 5---------------------
  # - h step--
  conv_r5 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X4)
  conv_r5 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r5)
  conv_r5 = Add()([X4,conv_r5])
  conv_r5 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r5)
  # - x step --
  # H^T(Hf-y)
  X5 = Lambda(GradientCASSI)([X4,y,H])
  X5_prior = Subtract()([X4,conv_r5])
  X5_prior = Mu_parameter()(X5_prior)

  X5 = Add()([X5_prior,X5])
  X5 = Lambda_parameter()(X5)
  X5 = Subtract(name='X5')([X4,X5])

          #--------Stage 6---------------------
  # - h step--
  conv_r6 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X5)
  conv_r6 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r6)
  conv_r6 = Add()([X5,conv_r6])
  conv_r6 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r6)
  # - x step --
  # H^T(Hf-y)
  X6 = Lambda(GradientCASSI)([X5,y,H])
  X6_prior = Subtract()([X5,conv_r6])
  X6_prior = Mu_parameter()(X6_prior)

  X6 = Add()([X6_prior,X6])
  X6 = Lambda_parameter()(X6)
  X6 = Subtract(name='X6')([X5,X6])

            #--------Stage 7---------------------
  # - h step--
  conv_r7 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X6)
  conv_r7 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r7)
  conv_r7 = Add()([X6,conv_r7])
  conv_r7 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r7)
  # - x step --
  # H^T(Hf-y)
  X7 = Lambda(GradientCASSI)([X6,y,H])
  X7_prior = Subtract()([X6,conv_r7])
  X7_prior = Mu_parameter()(X7_prior)

  X7 = Add()([X7_prior,X7])
  X7 = Lambda_parameter()(X7)
  X7 = Subtract(name='X7')([X6,X7])

              #--------Stage 8---------------------
  # - h step--
  conv_r8 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X7)
  conv_r8 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r8)
  conv_r8 = Add()([X7,conv_r8])
  conv_r8 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r8)
  # - x step --
  # H^T(Hf-y)
  X8 = Lambda(GradientCASSI)([X7,y,H])
  X8_prior = Subtract()([X7,conv_r8])
  X8_prior = Mu_parameter()(X8_prior)

  X8 = Add()([X8_prior,X8])
  X8 = Lambda_parameter()(X8)
  X8 = Subtract(name='X8')([X7,X8])

              #--------Stage 9---------------------
  # - h step--
  conv_r9 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X8)
  conv_r9 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r9)
  conv_r9 = Add()([X8,conv_r9])
  conv_r9 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r9)
  # - x step --
  # H^T(Hf-y)
  X9 = Lambda(GradientCASSI)([X8,y,H])
  X9_prior = Subtract()([X8,conv_r9])
  X9_prior = Mu_parameter()(X9_prior)

  X9 = Add()([X9_prior,X9])
  X9 = Lambda_parameter()(X9)
  X9 = Subtract(name='X9')([X8,X9])

  model = Model(inputs,X9)


  if (pretrained_weights):
      model.load_weights(pretrained_weights)
      print('loading weights generator')

  return model