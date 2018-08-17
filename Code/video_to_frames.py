'''
Using OpenCV takes a mp4 video and produces a number of images.
source : https://gist.github.com/keithweaver/70df4922fec74ea87405b83840b45d57
Requirements
----
You require OpenCV 3.2 to be installed.
Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py
Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

MSVD_VIDEO_DATA_DIR = "../Data/MSVD/YouTubeClips/"
MSVD_FRAMEOP_DIR = "../Data/MSVD/Frames"

def read_video_clips(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f))]

def vid_to_frame(fname,vid):
    DIR = MSVD_FRAMEOP_DIR+"/"+vid[:-4]
    try:
        if not os.path.exists(DIR):
            os.makedirs(DIR)
    except OSError:
        print ('Error: Creating directory of data')
    # Playing video from file:
    cap = cv2.VideoCapture(fname)
    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Saves image of the current frame in jpg file
        name = DIR+'/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        # To stop duplicate images
        currentFrame += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


try:
    if not os.path.exists(MSVD_FRAMEOP_DIR):
        os.makedirs(MSVD_FRAMEOP_DIR)
except OSError:
    print ('Error: Creating directory of data')

vid_clips_list = read_video_clips(MSVD_VIDEO_DATA_DIR)

for vid in vid_clips_list:
    fname = MSVD_VIDEO_DATA_DIR+vid
    vid_to_frame(fname,vid)