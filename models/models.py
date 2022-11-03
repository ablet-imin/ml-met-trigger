import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

#for plot with ATLAS styple
import matplotlib.pyplot as plt

def DNN(h_dim, out_dim,inputShape=None, activation='relu', L1L2=None,
        compile=False, optimizer=Adam(learning_rate=1e-3),
        loss='mean_squared_error', metrics=None ):
        
    model_nn = Sequential(name='DNN')
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
    
    if compile:
        model_nn.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics
                        )
    return model_nn
    

def model_CNN(h_dim, out_dim, inputShape, activation='relu',
              L1L2=None, compile=False, optimizer=Adam(learning_rate=1e-3),
              loss='mean_squared_error', metrics=None):
              
    _cnn = Sequential(name='convlayers')
    _cnn.add(tf.keras.layers.Conv2D(1, 2,
                                        strides=2,
                                        activation='relu')
            )
    _cnn.add(tf.keras.layers.Conv2D(3, 3,
                                    strides=2,
                                    activation='relu')
            )
    _cnn.add(tf.keras.layers.MaxPooling2D( (2, 2) ) )
    _cnn.add(tf.keras.layers.Flatten() )
    
    _dnn = DNN(h_dim=h_dim, out_dim=out_dim,
               activation=activation,
               L1L2=L1L2)
    #combine conv and dnn layers
    inputs = Input(shape=inputShape)
        
    _full_model = Model(
                    inputs=inputs,
                    outputs = _dnn(_cnn(inputs)),
                    name='CNN'
                    )
    #Compile models
    if compile:
        _full_model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics
                            )
    return _full_model

def model_combined(h_dim, out_dim, n_class, inputShape, model_type='cnn',
              output_type='regressor',activation='relu',
              L1L2=None, compile=False, optimizer=Adam(learning_rate=1e-3),
              loss='mean_squared_error', metrics=None):
              
    if model_type not in ['cnn', 'dnn']:
        raise Exception("model type: cnn or dnn")
    if output_type not in ['regressor', 'classifier', 'combined']:
        raise Exception("Chose: regressor, classifier or combined")
    
    _cnn = Sequential(name='convlayers')
    _cnn.add(tf.keras.layers.Conv2D(1, 2,
                                        strides=2,
                                        activation='relu')
            )
    _cnn.add(tf.keras.layers.Conv2D(3, 3,
                                    strides=2,
                                    activation='relu')
            )
    _cnn.add(tf.keras.layers.MaxPooling2D( (2, 2) ) )
    _cnn.add(tf.keras.layers.Flatten() )
    
    model_nn = Sequential(name='DNN')
    for node in h_dim:
        model_nn.add(
                     Dense(node,
                           activation = activation,
                           kernel_regularizer=L1L2,
                           bias_regularizer=L1L2
                           )
                     )
    
    inputs = Input(shape=inputShape)
    if model_type == 'dnn':
        out_dnn = model_nn(inputs)
    else:
        out_cnn = _cnn(inputs)
        out_dnn = model_nn(out_cnn)
    model_reg_output = Dense(out_dim,
                            #activation = 'linear',
                            #kernel_regularizer=L1L2,
                            #bias_regularizer=L1L2
                            name='regressor'
                            )(out_dnn)
    classifier_output = Dense(n_class,
                            activation = 'softmax',
                            #kernel_regularizer=L1L2,
                            #bias_regularizer=L1L2,
                            name='classifier'
                            )(out_dnn)
    #Full model
    output_model_list = []
    if output_type == 'combined':
        output_model_list = [model_reg_output, classifier_output]
    else:
        output_model_list = [model_reg_output] if output_type == 'regressor' else [classifier_output]
    
    _model = Model(
                    inputs=inputs,
                    outputs = output_model_list
                    )
    
    if compile:
        _model.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics
                        )
    return _model
              
            

