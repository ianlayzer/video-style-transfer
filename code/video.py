import cv2
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
from moviepy.editor import *
import numpy as np

frames_list = []
cap = cv2.VideoCapture('./../data/content/video/tomjerry.mp4')
i = 0
frame_skip = 150
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if i > frame_skip - 1:
        # cv2.imwrite('test_'+str(i)+'.jpg', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.show()
        frames_list.append(frame)
        i = 0
        continue
    i += 1

cap.release()

print(len(frames_list))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = VideoWriter('./../data/content/video/test.mp4', fourcc, 1, (640, 360))
for frame in frames_list:
    video.write(frame)
video.release()
cv2.destroyAllWindows()

# video = VideoFileClip("./../data/content/video/tomjerry.mp4")
# frames_iterable = video.iter_frames(fps=1)


# # frames_list = []
# # for frame in frames_iterable:
# #     plt.imshow(frame)
# #     plt.show()
# #     frame = np.asarray(frame, dtype=np.uint8)
# #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# #     print(frame.shape)
# #     frames_list.append(frame)



#######

# import cv2
# import numpy as np
# from moviepy.editor import *
# import matplotlib.pyplot as plt


# video = VideoFileClip("./../data/content/video/tomjerry.mp4")
# frames_iterable = video.iter_frames(fps=0.1)


# frames_list = []
# for frame in frames_iterable:
#     # plt.imshow(frame)
#     # plt.show()
#     frames_list.append(frame)

# print(len(frames_list))


# video = cv2.VideoCapture("./../data/content/video/tomjerry.mp4")
# i = 0
# # a variable to set how many frames you want to skip
# frame_skip = 100
# while video.isOpened():
#     ret, frame = video.read()
#     if not ret:
#         break
#     if i > frame_skip - 1:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         plt.imshow(frame)
#         plt.show()
#         i = 0
#         continue
#     i += 1

# video.release()
# cv2.destroyAllWindows()




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