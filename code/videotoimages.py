from moviepy.editor import *
import tensorflow as tf
# import matplotlib.pyplot as plt

# !! Before running this file, first run "pip install moviepy" !!
# (This file/the file this function is placed in needs to be run from the code directory to work)

def video_to_images(video_name, fps):
    video = VideoFileClip("./../data/content/video/" + video_name)

    frames_iterable = video.iter_frames(fps=fps)
    # This is not a normal python iterable, and can't really be used.
    # If we want an array of all of the images, we could use this loop to get each image
    #    and then add them one by one into an array (although this might not be necessary)
    for frame in frames_iterable:
        image = tf.convert_to_tensor(frame, dtype=tf.uint8)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Image is now ready for line 3 of preprocess_image in stylize (resizing)
        
        # plt.imshow(image.numpy())
        # plt.show()


# These can be read in as inputs from main
video_name = "tomjerry.mp4"
fps = 1

video_to_images(video_name, fps)






### (Ignore below) ###

# import cv2
# import os
# from moviepy.editor import *

# def video_to_images(video_name, framespersec):
#     # Converts the video to the desired fps, saves a new copy of the video
#     old_fps = VideoFileClip("./../data/content/video/" + video_name + ".mp4")
#     new_path = "./../data/content/video/" + video_name + "_" + str(framespersec) + "fps" + ".mp4"
#     old_fps.write_videofile(new_path, fps=framespersec, audio=False)
#     old_fps.reader.close()

#     video = cv2.VideoCapture(new_path)
#     currentFrame = 0
#     while(True):
#         # Capture frame-by-frame
#         ret, frame = video.read()

#         # Saves image of the current frame in jpg file
#         name = './../data/content/images/frame' + str(currentFrame) + '.jpg'
#         print ('Creating... ' + name)
#         cv2.imwrite(name, frame)

#         # To stop duplicate images
#         currentFrame += 1

#     # Releases the capture from memory
#     video.release()
#     cv2.destroyAllWindows()

# video_to_images(video_name, framespersec)