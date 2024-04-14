import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from tensorflow.keras import layers

img_x, img_y = (576, 576)
dx = 48
filelst = os.listdir('DRIVE/training/images/')
filelst = ['DRIVE/training/images/' + v for v in filelst]
imgs = [cv2.imread(file) for file in filelst]
filelst = os.listdir('DRIVE/training/1st_manual/')
filelst = ['DRIVE/training/1st_manual/' + v for v in filelst]
manuals = [np.asarray(Image.open(file)) for file in filelst]
imgs = [cv2.resize(v, (img_x, img_y)) for v in imgs]
manuals = [cv2.resize(v, (img_x, img_y)) for v in manuals]
X_train = np.array(imgs)
Y_train = np.array(manuals)
X_train = X_train.astype('float32') / 255.
Y_train = Y_train.astype('float32') / 255.
X_train = X_train[..., 1]  # the G channel
X_train = np.array([[X_train[:, v * dx:(v + 1) * dx, vv * dx:(vv + 1) * dx] for v in range(img_y // dx)] for vv in
                    range(img_x // dx)]).reshape(-1, dx, dx)[:, np.newaxis, ...]
Y_train = np.array([[Y_train[:, v * dx:(v + 1) * dx, vv * dx:(vv + 1) * dx] for v in range(img_y // dx)] for vv in
                    range(img_x // dx)]).reshape(-1, dx * dx)[..., np.newaxis]
temp = 1 - Y_train
Y_train = np.concatenate([Y_train, temp], axis=2)


def unet_model(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)

    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv6 = layers.Reshape((2, patch_height * patch_width))(conv6)
    conv6 = layers.Permute((2, 1))(conv6)

    conv7 = layers.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    return model


model = unet_model(X_train.shape[1], X_train.shape[2], X_train.shape[3])
model.summary()

checkpointer = ModelCheckpoint(filepath='best_weights.keras', verbose=1, monitor='val_acc',
                               mode='auto', save_best_only=True)
model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=2, shuffle=True, validation_split=0.2,
          callbacks=[checkpointer])

imgs = cv2.imread('DRIVE/test/images/01_test.tif')[..., 1]  # the G channel
imgs = cv2.resize(imgs, (img_x, img_y))
manuals = np.asarray(Image.open('DRIVE/test/1st_manual/01_manual1.gif'))
X_test = imgs.astype('float32') / 255.
Y_test = manuals.astype('float32') / 255.
X_test = np.array([[X_test[v * dx:(v + 1) * dx, vv * dx:(vv + 1) * dx] for v in range(img_y // dx)] for vv in
                   range(img_x // dx)]).reshape(-1, dx, dx)[:, np.newaxis, ...]
model.load_weights('best_weights.h5')
Y_pred = model.predict(X_test)
Y_pred = Y_pred[..., 0].reshape(img_x // dx, img_y // dx, dx, dx)
Y_pred = [Y_pred[:, v, ...] for v in range(img_x // dx)]
Y_pred = np.concatenate(np.concatenate(Y_pred, axis=1), axis=1)
Y_pred = cv2.resize(Y_pred, (Y_test.shape[1], Y_test.shape[0]))
plt.figure(figsize=(6, 6))
plt.imshow(Y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(Y_test)
