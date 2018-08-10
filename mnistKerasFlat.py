'''
In this program, we use flat deep neural network and Keras framework to train NN to classify MNIST dataset
We use 1,000 samples in a batch of 100 to train the NN and plot performance againts each iteration
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# numpy package
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# number of samples
m = x_train.shape[0]
# image dimensions (assumed square)
h = x_train.shape[1]
w = x_train.shape[2]
input_size = h * w
# flatten image and rescale it (m,h*w)
x_train = np.reshape(x_train,[-1, h*w]).astype('float32')/255
x_test = np.reshape(x_test,[-1, h*w]).astype('float32')/255

# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45
num_labels=10

# this is 3-layer MLP with ReLU and dropout after each layer
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
# this is the output for one-hot vector
model.add(Activation('softmax'))
model.summary()

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train model
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# validate the model 
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))