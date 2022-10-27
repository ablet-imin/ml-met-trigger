import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

#for plot with ATLAS styple
import matplotlib.pyplot as plt

def DNN(h_dim, out_dim, activation='relu', L1L2='None' ):
    model_nn = Sequential(name='DNN')
    regularizers = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)
    for node in h_dim:
        model_nn.add(
                     Dense(node,
                           activation = activation,
                           kernel_regularizer=L1L2,
                           bias_regularizer=L1L2
                           )
                     )
    #outpu layer
    model_nn.add(Dense(out_dim,
                        activation = 'linear',
                       #kernel_regularizer=L1L2,
                       #bias_regularizer=L1L2
                      )
                )
    return model_nn
    
    

