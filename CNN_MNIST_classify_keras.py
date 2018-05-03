import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# shape of x_train: (60000, 28, 28)
# shape of x_test: (10000, 28, 28)
# shape of y_train: (60000,)
# shape of y_test: (10000,)
# data type of these arrays is uint8 - unsigned integer between 0 and 255

img_rows = 28
img_cols = 28

# Since we already know that these images have no channel info, using 1 for n_channels
if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


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
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=n_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# loss='categorical_crossentropy' could also be used
# optimizer='Adadelta' could also be used
# Default parameters of the optimizer will be used.

batch_size = 128
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])