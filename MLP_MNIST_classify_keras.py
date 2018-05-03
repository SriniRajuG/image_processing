'''
Use a fully-connected neural network (Multi layer perceptron) to classify the MNIST images.
https://en.wikipedia.org/wiki/MNIST_database

TODO
1) Use metrics from sk-learn to calculate the confusion matrix
'''

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras import backend

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# shape of x_train: (60000, 28, 28)
# shape of x_test: (10000, 28, 28)
# shape of y_train: (60000,)
# shape of y_test: (10000,)
# data type of these arrays is uint8 - unsigned integer between 0 and 255


img_rows = x_test.shape[1]
img_cols = x_test.shape[2]
img_vec_dim = img_rows * img_cols
x_train = x_train.reshape(-1, img_vec_dim)
x_test = x_test.reshape(-1, img_vec_dim)
# This shape is specific to MLP.
# For CNN, shape is (img_rows, img_cols, n_channels) or (n_channels, img_rows, img_cols)

# normalize arrays
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # image array values unsigned integers between 0 and 255
x_test /= 255

n_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, n_classes)
# assumes that classes start at 0 and not 1
y_test = keras.utils.to_categorical(y_test, n_classes)


model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(img_vec_dim,)))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
#------
# When using the categorical_crossentropy loss, your targets should be in categorical format
# (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that
# is all-zeros expect for a 1 at the index corresponding to the class of the sample).
# In order to convert integer targets into categorical targets, you can use the Keras utility
# to_categorical:

# String (name of optimizer) or optimizer object can be used for the parameter optimizer

# currently Keras doesn't have F1, precision and recall as metrics. So use sk-learn to calculate
# these metrics
#----------

batch_size = 128
epochs = 20
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                    validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
