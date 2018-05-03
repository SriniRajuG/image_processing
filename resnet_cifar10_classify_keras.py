import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

N_BLOCKS = 3
N_CNN_PER_SUB_BLOCK = 2
KERNEL_SIZE = 3


def cnn_block(inputs, num_filters=16, kernel_size=KERNEL_SIZE, strides=1, activation='relu',
              batch_normalization=True, conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        conv_first (bool): conv-bn-activation (True) or activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    x = inputs
    if conv_first:
        x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = Activation('relu')(x)
        x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        return x


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    The number of filters doubles when the feature maps size
    is halved.
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    global N_BLOCKS
    global N_CNN_PER_SUB_BLOCK
    if (depth - 2) % (N_CNN_PER_SUB_BLOCK * N_BLOCKS) != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    inputs = Input(shape=input_shape)
    num_filters = 16
    n_sub_blocks = int((depth - 2) / (N_BLOCKS * N_CNN_PER_SUB_BLOCK))

    x = cnn_block(inputs=inputs)
    # Instantiate convolutional base (stack of blocks).
    for i in range(N_BLOCKS):
        for j in range(n_sub_blocks):
            if j == 0 and i > 0:
                is_first_layer_but_not_first_block = True
                strides = 2
            else:
                is_first_layer_but_not_first_block = False
                strides = 1
            y = cnn_block(inputs=x, num_filters=num_filters, strides=strides)
            y = cnn_block(inputs=y, num_filters=num_filters, activation=None)
            if is_first_layer_but_not_first_block:
                x = cnn_block(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides,
                              activation=None, batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    # final block's output image is 8*8 hence pool-_size in average_pooling is 8
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)

# normalize arrays
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # image array values unsigned integers between 0 and 255
x_test /= 255
input_shape = x_train.shape[1:]

subtract_pixel_mean = True
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

n_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, n_classes)
# assumes that classes start at 0 and not 1
y_test = keras.utils.to_categorical(y_test, n_classes)

n_sub_blocks = 3
# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1
# Computed depth from supplied model parameter n
if version == 1:
    depth = (N_BLOCKS * N_CNN_PER_SUB_BLOCK * n_sub_blocks) + 2
elif version == 2:
    depth = n_sub_blocks * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)


if version == 2:
    # model = resnet_v2(input_shape=input_shape, depth=depth)
    pass
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)


# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]


data_augmentation = True
batch_size = 32
epochs = 1
# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True)
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

