import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

def tf_sqrt(X, axis=-1):
    Y = tf.math.square(X)
    Y = tf.math.reduce_sum(Y, axis=axis, keepdims=True)
    return tf.math.sqrt(Y)

class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        met_true = tf_sqrt(y_true, 1)
        met_pred = tf_sqrt(y_pred, 1)
        met_loss = tf.keras.losses.MeanAbsolutePercentageError()(met_true, met_pred)
        mse_loss_x = tf.keras.losses.MeanSquaredError()(y_true[:,0], y_pred[:,0])
        mse_loss_y = tf.keras.losses.MeanSquaredError()(y_true[:,1], y_pred[:,1])
        return mse_loss_x+mse_loss_y+met_loss

    
class MSPE_ET(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        relative = tf.math.square((y_true-y_pred))
        relative = tf.divide(relative,y_true)
        _loss = tf.reduce_mean(relative, axis=-1)
        return _loss
class MeanSquaredError_ET(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        met_MSPE = MSPE_ET()(y_true, y_pred)
        mse_MSE = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        return 0.5*met_MSPE+0.5*mse_MSE

