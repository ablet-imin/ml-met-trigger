import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

def tf_sqrt(X, axis=-1):
    Y = tf.math.square(X)
    Y = tf.math.reduce_sum(Y, axis=axis, keepdims=True)
    return tf.math.sqrt(Y)


    
class MSPE_ET(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):
        relative = tf.math.square((y_true-y_pred))
        relative = tf.math.divide(relative,tf.math.abs(y_true))
        _loss = tf.reduce_sum(relative)
        return _loss


