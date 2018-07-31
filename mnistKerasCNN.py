
'''
Deep Network (CNN) using Keras to Predict MNIST digits 
Program domestrate how to train CNN using Keras/Tensorflow predict outcomes
Training Set Data        ---  60000 images of digits with 28,28,1 dimensions 
Training Labels          ---  digits between 0 through 9

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 28, 28, 1)         0         
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 34, 34, 1)         0         
_________________________________________________________________
conv0 (Conv2D)               (None, 32, 32, 32)        320       
_________________________________________________________________
bn0 (BatchNormalization)     (None, 32, 32, 32)        128       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
max_pool (MaxPooling2D)      (None, 16, 16, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0         
_________________________________________________________________
fc1 (Dense)                  (None, 128)               1048704   
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
fc2 (Dense)                  (None, 10)                1290      
=================================================================
Total params: 1,050,442
Trainable params: 1,050,378
Non-trainable params: 64

'''


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')


# define seed to reproduce results
seed = 7
np.random.seed(seed)

# load data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be m,n_h,n_w,n_c structure
X_train = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# update labels to one hot encode outputs instead of digits
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# Define Layers

def mnistModel(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
   
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    X = Dropout(0.2)(X)
    X = Dense(10, activation='sigmoid', name='fc2')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='mnistModel')

    return model

# Define Model

mnistModel = mnistModel((28,28,1))

# Compile Model

mnistModel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# Train Model

mnistModel.fit(x =X_train, y = y_train, batch_size=16, epochs=40)

