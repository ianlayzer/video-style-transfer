# import cv2
# import matplotlib.pyplot as plt
# from cv2 import VideoWriter, VideoWriter_fourcc
# import numpy as np

# # def get_frames():
# frames_list = []
# cap = cv2.VideoCapture('./../data/content/video/tomjerry.mp4')
# i = 0
# # a variable to set how many frames you want to skip
# frame_skip = 100
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     if i > frame_skip - 1:
#         # cv2.imwrite('test_'+str(i)+'.jpg', frame)
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         plt.imshow(frame)
#         plt.show()
#         frames_list.append(frame)
#         i = 0
#         continue
#     i += 1

# cap.release()
# # cv2.destroyAllWindows()

# # get_frames()

# # for frame in frames_list:
# #     print(frame.dtype)

# # fourcc = VideoWriter_fourcc(*'mp4v')
# # video = VideoWriter('./../data/content/video/test.mp4', fourcc, 1, (360, 640))
# # for frame in frames_list:
# #     video.write(np.asarray(frame, np.uint8))
# # video.release()



#######

import cv2
import numpy as np
from moviepy.editor import *
import matplotlib.pyplot as plt


video = VideoFileClip("./../data/content/video/tomjerry.mp4")
frames_iterable = video.iter_frames(fps=0.1)


# frames_list = []
# for frame in frames_iterable:
#     # plt.imshow(frame)
#     # plt.show()
#     frames_list.append(frame)

# print(len(frames_list))


video = cv2.VideoCapture("./../data/content/video/tomjerry.mp4")
i = 0
# a variable to set how many frames you want to skip
frame_skip = 100
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    if i > frame_skip - 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.show()
        i = 0
        continue
    i += 1

video.release()
cv2.destroyAllWindows()




##########

# def preprocess_video(video_name):
# 	frame_list = []
# 	video = cv2.VideoCapture("./../data/content/video/" + video_name)
# 	i = 0
#     # a variable to set how many frames you want to skip
# 	frame_skip = 100
# 	while video.isOpened():
# 		ret, frame = video.read()
# 		if not ret:
# 			break
# 		if i > frame_skip - 1:
# 			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 			frame_list.append(preprocess_frame(frame))
# 			i = 0
# 			continue
# 		i += 1

# 	video.release()

# 	return frame_list



###########
# def preprocess_video(video_name):
# 	# get video
# 	video = VideoFileClip("./../data/content/video/" + video_name)
# 	frames_iterable = video.iter_frames(fps=0.1)

# 	# preprocess and add each frame in frame iterable to python list for indexing
# 	frame_list = []
# 	for frame in frames_iterable:
# 		frame_list.append(preprocess_frame(frame))
# 	return frame_list