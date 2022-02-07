import keras
import tensorflow as tf
from keras import applications, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.python.keras.regularizers import l2

tf.config.experimental.set_visible_devices([], 'GPU')
data_dir = "D:/FinalProject/RNN/binary_rnn_data"
data_save_path_x = "D:/FinalProject/RNN/rnn-BinaryData/Rnn_Sub_Data_X_Ver4.npy"   # TODO: Change path each session
data_save_path_y = "D:/FinalProject/RNN/rnn-BinaryData/Rnn_Sub_Data_Y_Ver4.npy"   # TODO: Change path each session
seq_len = 20                         # TODO: the size of video array is 10 (seq_len\4 for sub) ~~ 7 FPS  (20 used)
classes = ["FistVert", "HeadTurn"]       # "FistHorz"
pixel_size = 64
known_Y = True
frameRate = 1
new_data = True
threshold_value = 30
#version_name = 'Rnn_Sub_Ver2'           # TODO: Change version number each time
# known_Y = True if test or train


#  Creating frames from videos
def frames_extraction(video_path):
    frames_list = []
    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable
    count = 1

    while count != seq_len+1:
        success, image = vidObj.read()
        if success and (count % frameRate) == 0:
            image = cv2.resize(image, (pixel_size, pixel_size))
            frames_list.append(image)
            count += 1
        else:
            count += 1
        """
        else:
            print("Defected frame")
            break
        if len(frames_list) != seq_len/2:
        iterations = int(seq_len/2 - len(frames_list))
        for i in range(1, iterations):
            frames_list.append(image)
        """
    return frames_list


def sub_frames(frames_list):
    frame_index = 0
    sub_array = []
    while frame_index < seq_len-1:
        # Image Sub
        result = cv2.subtract(frames_list[frame_index + 1], frames_list[frame_index])
        grey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Image to Binary + threshold
        ret, bw_img = cv2.threshold(grey, threshold_value, pixel_size, cv2.THRESH_BINARY)
        # CCA
        output = cv2.connectedComponentsWithStats(bw_img, 4, cv2.CV_32S)  # Can assign 8 instead of 4 - cant see difference
        (numLabels, labels, stats, centroids) = output
        max_cc = [0, 0, 0, 0, 0, 0]  # CC representation --> (index,area,x,y,w,h)
        """
        Info on Stats operation:
        numLabels = number of total components
        labels = A mask named labels has the same spatial dimensions as our input thresh image. 
          For each location in labels, we have an integer ID value that corresponds to the connected component where the pixel belongs.
        stats = Statistics on each connected component, including the bounding box coordinates and area (in pixels).
        centroids = (x, y)-coordinates of each connected component.
        """
        # Iterate on CC of result(the sub from the 2 images)
        for i in range(0, numLabels):
            if i > 0:
                # text = "examining component {}/{}".format(i + 1, numLabels)
                # print("[INFO] {}".format(text))
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                (cX, cY) = centroids[i]
                # Update max CC
                if area > max_cc[1]:
                    max_cc[0] = i
                    max_cc[1] = area
                    max_cc[1] = x
                    max_cc[1] = y
                    max_cc[1] = w
                    max_cc[1] = h
                    # Bounding box surrounding the connected component along with
                    output = result.copy()
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, pixel_size, 0), 3)
                    cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, pixel_size), -1)
                    # Construct a mask for the current connected component b finding a pixels in the labels array that have the current CC id
                    componentMask = (labels == i).astype("uint8") * pixel_size
                    max_cc_img = componentMask
                    # show our output image and connected component mask
                else:
                    continue
            else:
                max_cc_img = bw_img  # TODO: Check effect - meaning no CC so save the original image
        #  Save the CC we want to use on folder
        max_cc_img = np.expand_dims(max_cc_img, axis=2)
        sub_array.append(max_cc_img)
        frame_index = frame_index + 2
    return sub_array


def create_data(input_dir, known_Y):
    X = []
    if known_Y:
        Y = []
    if known_Y:
        classes_list = os.listdir(input_dir)
        exists1 = ".DS_Store" in classes_list
        if exists1:
            classes_list.remove(".DS_Store")
        for c in classes_list:
            print(c)
            files_list = os.listdir(os.path.join(input_dir, c))
            exists2 = ".DS_Store" in files_list
            if exists2:
                files_list.remove(".DS_Store")
            for f in files_list:
                frames1 = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
                if len(frames1) == seq_len:
                    frames = sub_frames(frames1)
                    X.append(frames)
                    y = [0] * len(classes)
                    y[classes.index(c)] = 1
                    Y.append(y)

    else:
        files_list = os.listdir(input_dir)
        exists2 = ".DS_Store" in files_list
        if exists2:
            files_list.remove(".DS_Store")
        for f in files_list:
            frames = frames_extraction(os.path.join(input_dir, f))
            if len(frames) == seq_len / 2:
                X.append(frames)
                # TODO: Add video-py code to make another video

    X = np.asarray(X)

    if not known_Y:
        return X, files_list
    else:
        Y = np.asarray(Y)
        return X, Y


if new_data:
    X, Y = create_data(data_dir, known_Y)
    np.save(data_save_path_x, X)
    np.save(data_save_path_y, Y)

"""X, Y = create_data(data_dir, known_Y)
data_save_path_x = "D:/FinalProject/RNN/binary_rnn_data/Rnn_Sub_Data_X_New.npy"
data_save_path_y = "D:/FinalProject/RNN/binary_rnn_data/Rnn_Sub_Data_X_New.npy"
np.save(data_save_path_x, X)
np.save(data_save_path_y, Y)"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True, data_format='channels_first',
                     recurrent_activation='hard_sigmoid', activation='tanh', padding='same', input_shape=(int(seq_len / 2), pixel_size, pixel_size, 1)))
#model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first'))  #or pool_size(1,3,3)
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)
                        , data_format='channels_first'
                       , padding='same', return_sequences=True))
model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dense(2, activation="sigmoid"))   # TODO: num of classes + try sigmoid
model.summary()
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])  # maybe loss='categorical_crossentropy'w
earlystop = EarlyStopping(monitor="val_loss", patience=7)
callbacks = [earlystop]
history = model.fit(x=X_train, y=y_train, epochs=500, batch_size=16, shuffle=True, validation_split=0.2, callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
model.save('bestModel.h5')









