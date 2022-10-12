#!/usr/bin/env python
# coding: utf-8

import os, sys
import numpy as np
import pandas as pd
import seaborn as sns

#for plot with ATLAS styple
import matplotlib.pyplot as plt
from matplotlib import colors

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import uproot
import gc

from cells import Cells 
from models import DNN

MCdata="cell_analysis_mu60"
MC_root = f"../re21.9/{MCdata}/cells.root"
tree_name = "ntuple"
        
x_bin = np.linspace(-3.15, 3.15, num=33)
y_bin = np.linspace(-5, 5, num=51)

cell_data = Cells(MC_root, unit="GeV")
cell_imgs, cell_label = cell_data.cimg(x_bin, y_bin, batch_size=500)

print(f"Cell image shape: {cell_imgs.shape}")
print(f"Image label shape: {cell_label.shape}")

gc.collect()



test_indx = -1000
x_train, y_train = cell_imgs[:test_indx,:,:,:].reshape((-1,50*32*2)), cell_label[:test_indx,:]
x_test, y_test = cell_imgs[test_indx :,:,:,:].reshape(-1,50*32*2), cell_label[test_indx :,:]

def threshold_zero(data, thold=0.5):
    data[abs(data)<thold ] = 0


print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

from sklearn.utils import shuffle
x_train,  y_train =   shuffle(x_train, y_train, random_state=0)

gc.collect()


# # ML Model


from tensorflow.keras.layers import Layer, GaussianNoise
from tensorflow.keras import backend as K

def tf_sqrt(X, axis=-1):
    Y = tf.math.square(X)
    Y = tf.math.reduce_sum(Y, axis=axis, keepdims=True)
    return tf.math.sqrt(Y)

class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        met_true = tf_sqrt(y_true, 1)
        met_pred = tf_sqrt(y_pred, 1)
        met_loss = tf.keras.losses.MeanSquaredError()(met_true, met_pred)
        mse_loss_x = tf.keras.losses.MeanSquaredError()(y_true[:,0], y_pred[:,0])
        mse_loss_y = tf.keras.losses.MeanSquaredError()(y_true[:,1], y_pred[:,1])
        return mse_loss_x+mse_loss_y+met_loss



model = DNN(x_train.shape[-1:], [2000], 2, L1L2='l1_l2')

model.compile(optimizer=Adam(learning_rate=5e-3), loss=MeanSquaredError(), metrics=['mean_absolute_error'])

gc.collect()

tf.keras.utils.plot_model(model, expand_nested=True)




history = model.fit(x_train, y_train, 
                    batch_size= 1024,
                    epochs=5, 
                    verbose=1, validation_split=0.15)



gc.collect()


def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0.01, 15000])
        #plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Error ')
        plt.legend()
        plt.grid(True)
        
fig = plt.figure(figsize =(10, 10))
plot_loss(history)


# In[31]:


predicted = model.predict(x_test)

def np_met(X, axis=1):
    return np.sqrt( np.sum(np.square(X), axis) )

predicted_met = np_met(predicted)
true_met = np_met(y_test)

fig = plt.figure(figsize =(10, 10))
plt.scatter(np.squeeze(true_met),np.squeeze(predicted_met), marker='o', alpha=0.7)
plt.xlabel("truth met")
plt.ylabel("predicted met")
plt



# In[33]:

fig = plt.figure(figsize =(10, 10))
_,bins,_ = plt.hist(predicted[:,1], bins=100,alpha=0.7, color='red')
_ = plt.hist(y_test[:,1], bins=100, alpha=0.5, histtype='step', color='Blue')


# In[34]:

fig = plt.figure(figsize =(10, 10))
_,bins,_ = plt.hist(predicted_met, bins=100,alpha=0.7, color='red')
_ = plt.hist(true_met, bins=100, alpha=0.5, histtype='step', color='Blue')


# In[35]:

fig = plt.figure(figsize =(10, 10))
delta = true_met-predicted_met
delta.shape
gc.collect()
y_test.shape
#gc.collect()
_ = plt.hist(delta, bins=100)





