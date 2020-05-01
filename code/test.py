import cv2
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np

frames_list = []
cap = cv2.VideoCapture('../data/content/video/tomjerry.mp4')
i = 0
frame_skip = 100
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if i > frame_skip - 1:
        # cv2.imwrite('test_'+str(i)+'.jpg', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(frame)
        # plt.show()
        frames_list.append(frame)
        i = 0
        continue
    i += 1

cap.release()

fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter('../data/content/video/test.mp4', fourcc, 1, (360, 640))
for frame in frames_list:
    video.write(np.asarray(frame, np.uint8))
video.release()