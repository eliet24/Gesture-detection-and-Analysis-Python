import numpy as np
import cv2
import matplotlib.pyplot as plt
#matplotlib inline  # if you are running this code in Jupyter notebook
seq_len = 20                          # TODO: the size of video array is 10 (seq_len\4 for sub) ~~ 7 FPS  (20 used)
classes = ["FistVert", "HeadTurn"]       # "FistHorz"
pixel_size = 64
known_Y = True
frameRate = 1
new_data = True
threshold_value = 30


img1 = cv2.imread('C:/Users/eliet/OneDrive/Desktop/New folder - Copy/subtracte5.jpg', 0)
#img1 = cv2.resize(img1, (pixel_size, pixel_size))
ret, bw_img = cv2.threshold(img1, threshold_value, pixel_size, cv2.THRESH_BINARY)
cv2.imwrite("C:/Users/eliet/OneDrive/Desktop/New folder - Copy/binary55.jpg", bw_img)