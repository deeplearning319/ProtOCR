import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, BatchNormalization, Flatten, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
%matplotlib inline
K.set_image_data_format('channels_last')

letters=loadmat('drive/Datasets/emnist-letters.mat')

data = letters['dataset']

X_train_orig = data[0][0][0][0][0][0].reshape(124800,28,28)
Y_train = to_categorical(data[0][0][0][0][0][1]-1)
X_test_orig = data[0][0][1][0][0][0].reshape(20800,28,28)
Y_test = to_categorical(data[0][0][1][0][0][1]-1)
"""print('X_train_orig shape: ' + str(X_train_orig.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test_orig shape:  ' + str(X_test_orig.shape))
print('Y_test shape:  ' + str(Y_test.shape))"""

import cv2
def transform(img):
    flipped_img = cv2.flip(img,1)
    rows,cols = flipped_img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(flipped_img,M,(cols,rows))
    return dst

X_train = np.zeros((124800, 28, 28), dtype = np.uint8)
for img,i in zip(X_train_orig,range(124800)):
    X_train[i] = transform(img)
X_test = np.zeros((20800, 28, 28), dtype = np.uint8)
for img,i in zip(X_test_orig,range(20800)):
    X_test[i] = transform(img)
X_train = X_train.reshape(124800, 28, 28,1)
X_test = X_test.reshape(20800, 28, 28,1)
"""print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape:  ' + str(X_test.shape))
print('Y_test shape:  ' + str(Y_test.shape))"""

def create_model_v5(input_shape):
    X_input = Input(shape = input_shape, name = 'input')
    X = Conv2D(filters = 64, kernel_size = [3, 3], padding = 'same', name = 'conv0a')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0a')(X)
    X = Activation('elu',name = 'act0a')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    X = Conv2D(filters = 128, kernel_size = [3, 3], padding = 'valid', name = 'conv0b')(X)
    X = BatchNormalization(axis = 3, name = 'bn0b')(X)
    X = Activation('elu',name = 'act0b')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    X = Conv2D(filters = 256, kernel_size = [3, 3], padding = 'valid', name = 'conv0c')(X)
    X = BatchNormalization(axis = 3, name = 'bn0c')(X)
    X = Activation('elu',name = 'act0c')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    X = Dropout(rate = 0.2)(X)
    X = Flatten()(X)
    X = Dense(350, activation = 'elu', name = 'fc0')(X)
    X = Dense(200, activation = 'elu', name = 'fc1')(X)
    X = Dense(100, activation = 'elu', name = 'fc2')(X)
    X = Dense(26, activation='softmax', name='fc3')(X)
    model = Model(inputs = X_input, outputs = X, name='Model')
    return model

model_v5 = create_model_v5(X_train.shape[1:])
model_v5.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpointer = ModelCheckpoint('model_weights.h5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(patience=2, verbose=1)
model_v5.fit(x = X_train/255., y = Y_train, epochs = 15, batch_size = 32, validation_data = (X_test/255., Y_test),callbacks=[checkpointer,earlystopper])