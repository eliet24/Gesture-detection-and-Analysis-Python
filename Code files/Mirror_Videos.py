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
 my_new_clip = my_clip.fx(vfx.mirror_x)
 my_new_clip.write_videofile("mir_fist" + str(i)+".mp4")


def main():
    i=0
    for file in os.listdir("C:/Users/eliet/OneDrive/Desktop/NewFistVert"):  #enter video files directory
     if file.endswith(".mp4"):
        path = os.path.join("C:/Users/eliet/OneDrive/Desktop/NewFistVert", file)  #same directory here
        dup_video(path, i)
        i = i+1

main()
