import cv2
import os
from moviepy.editor import *

# Before running this file, first run "pip install moviepy"

# these can be read in as inputs from main
video_name = "tomjerry"
framespersec = 1

def video_to_images(video_name, framespersec):
    old_fps = VideoFileClip("./../data/content/video/" + video_name + ".mp4")
    new_path = "./../data/content/video/" + video_name + "_" + str(framespersec) + "fps" + ".mp4"
    old_fps.write_videofile(new_path, fps=framespersec)
    old_fps.reader.close()

    video = cv2.VideoCapture(new_path)
    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = video.read()

        # Saves image of the current frame in jpg file
        name = './../data/content/images/frame' + str(currentFrame) + '.jpg'
        print ('Creating... ' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # Releases the capture from memory
    video.release()
    cv2.destroyAllWindows()

video_to_images(video_name, framespersec)