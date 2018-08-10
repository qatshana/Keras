# Keras-MNIST

This repo domestrate how to use Keras on MNIST using flat deep neural network and CNN

# Flat DNN Performance

Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 256)               200960    
_________________________________________________________________
activation_1 (Activation)    (None, 256)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               65792     
_________________________________________________________________
activation_2 (Activation)    (None, 256)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                2570      
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0


We achieved 98.30% accuracy on training set and .0529 loss on same set. We acheived 98.1% on test set (we used dropout/regularization) to minimize overfitting

# CNN Performance
