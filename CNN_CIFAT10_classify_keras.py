#TODO
# Create and use validation set instead of using test set for validation


import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)

n_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, n_classes)
# assumes that classes start at 0 and not 1
y_test = keras.utils.to_categorical(y_test, n_classes)

# normalize arrays
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # image array values unsigned integers between 0 and 255
x_test /= 255


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

data_augmentation = True
batch_size = 32
epochs = 1

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'CNN_cifar10_keras_trained_model.h5'

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do realtime data augmentation:
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True)
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                        validation_data=(x_test, y_test), workers=4)



# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



