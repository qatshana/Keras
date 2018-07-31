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
from keras.models import model_from_json
get_ipython().run_line_magic('matplotlib', 'inline')


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

#verify output
print(X_train.shape)

# Load model


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# run using test data
scores = loaded_model.evaluate(X_train, y_train, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

# run using test data
scores = loaded_model.evaluate(X_test, y_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

#plot_model(loaded_model, to_file='mnistModel.png')
#SVG(model_to_dot(loaded_model).create(prog='dot', format='svg'))