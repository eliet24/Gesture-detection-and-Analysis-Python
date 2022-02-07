import os
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, Sequential

tf.config.experimental.set_visible_devices([], 'GPU')
seq_len = 30                            # TODO: the size of video array is 10 (seq_len\4 for sub) ~~ 7 FPS
classes = ["FistVert", "HeadTurn"]     # "FistHorz"
pixel_size = 64
known_Y = True
frameRate = 1
threshold_value = 30


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
            frames1 = frames_extraction(os.path.join(input_dir, f))
            if len(frames1) == seq_len:
                frames = sub_frames(frames1)
                X.append(frames)
    X = np.asarray(X)
    if not known_Y:
        return X, files_list
    else:
        Y = np.asarray(Y)
        return X, Y


# identical to the previous one

model_path = "D:/Users/eliet/PycharmProjects/untitled/bestModel_frames_num=30.h5"
model = load_model(model_path)
# For the undecidable videos of the Mother code
data_dir = "C:/Users/eliet/OneDrive/Desktop/HeadTest"
X_unknown, files_list = create_data(data_dir, False)
y_pred_unknown = model.predict(X_unknown)
precentage=y_pred_unknown
sum=0
y_pred_unknown = np.argmax(y_pred_unknown, axis=1)

for i in range(len(files_list)):
    print("X=%s, Predicted=%s, predict=%s" % (files_list[i], y_pred_unknown[i], precentage[i]))

    if (y_pred_unknown[i]==1):
        sum=sum+1
print(sum/105)

