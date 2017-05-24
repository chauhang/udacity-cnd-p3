import keras
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU, Conv2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint


# NVidia model used
def build_model(rows, cols, channels):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(rows, cols, channels)))
    #model.add(Conv2D(3, (1, 1), padding='same', name='color_conv'))
    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2), kernel_initializer="he_normal", name="Conv1", padding="valid"))
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2), kernel_initializer="he_normal", name="Conv2", padding="valid"))
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2), kernel_initializer="he_normal", name="Conv3", padding="valid"))
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1), kernel_initializer="he_normal", name="Conv4", padding="valid"))
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1), kernel_initializer="he_normal", name="Conv5", padding="valid"))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu', kernel_initializer="he_normal",  name='FC1'))
    model.add(Dense(100, activation='elu',kernel_initializer="he_normal",  name='FC2'))
    model.add(Dense(50, activation='elu', kernel_initializer="he_normal", name='FC3'))
    model.add(Dense(10, activation='elu', kernel_initializer="he_normal", name='FC4'))
    model.add(Dense(1, kernel_initializer="he_normal", name='output'))
    model.summary()


    #model = make_parallel(model, NUM_GPUS)

    # compile
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='mse', metrics=[])
    
    return model