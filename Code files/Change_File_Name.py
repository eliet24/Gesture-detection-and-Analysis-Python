import moviepy
import cv2
import os
import moviepy.video.fx.all as vfx
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.editor as mpy
from moviepy.video.fx.all import crop
from moviepy.editor import *


def dup_video(file, i):
    my_clip = VideoFileClip(file)
    my_clip = my_clip.set_fps(30)
    my_clip.write_videofile("new_video" + str(i)+".mp4")


def main():
    i=0
    for file in os.listdir("D:/FinalProject/RNN/rnn-videos/New_Head_right_videos"):  #enter video files directory
     if file.endswith(".mp4"):
        path = os.path.join("D:/FinalProject/RNN/rnn-videos/New_Head_right_videos", file)  #same directory here
        dup_video(path, i)
        i = i+1

main()