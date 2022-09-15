import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

#for plot with ATLAS styple
import matplotlib.pyplot as plt

class NN():
    def __init__(self, layers, activation='relu', learning_rate=0.01, cnn=None ):
        self.layers = layers #number of nodes in each layers
        self.h_dimention = len(self.layers)
        self.activation = activation
        self.cnn = cnn
        self.model = self.buil_model(learning_rate=learning_rate)
        
        
    def buil_model(self, learning_rate):
        h_layers = []
        _model = tf.keras.Sequential()
        if self.cnn:
            _model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
            _model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
            _model.add(MaxPooling1D(pool_size=2))
            _model.add(Flatten())
            
        for i in self.layers:
            _model.add(Dense(i, activation=self.activation))
        
        #output layer
        _model.add(tf.keras.layers.Dense(1))
                
        _model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])
        return _model
        
    def plot_loss(self,history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0.001, 1000])
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Error ')
        plt.legend()
        plt.grid(True)
        
    def train(self,x_train, y_train, epochs, batch_size=128, save_interval=50):
        history  = self.model.fit(x_train, y_train, epochs=epochs, verbose=0,
                       validation_split = 0.3)
        self.plot_loss(history)
        
        return history
