from moviepy.editor import *
import tensorflow as tf
import numpy as np
import cv2
from pyflow import demo
import skimage

# This 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import matplotlib.pyplot as plt

# !! Before running this file, first run "pip install moviepy" !!
# (Note: you have to cd-ed into the code directory for this function to work)

def video_to_images(video_name, fps):
	video = VideoFileClip("./../data/content/video/" + video_name)

	frames_iterable = video.iter_frames(fps=fps)
	# This is not a normal python iterable, and can't really be used.
	# If we want an array of all of the images, we could use this loop to get each image
	#    and then add them one by one into an array (although this might not be necessary)
	num_frames = 0

	image_height = 300
	image_width = 600

	frame_list = []

	for frame in frames_iterable:
		frame_list.append(frame)

	# print(len(frame_list))

	new_list = []
	for i in range(40):
		new_list.append(frame_list[i])

	curr_image = new_list[0]
	curr_image = tf.convert_to_tensor(frame, dtype=tf.uint8)
	curr_image = tf.image.convert_image_dtype(curr_image, tf.float32)
	curr_image = tf.image.resize(curr_image, (image_height, image_width), antialias=True)
	curr_image = curr_image.numpy()
	prev_image = curr_image

	for frame in new_list:
		# if num_frames > 1:
		# 	break

		num_frames += 1
		next_image = tf.convert_to_tensor(frame, dtype=tf.uint8)
		next_image = tf.image.convert_image_dtype(next_image, tf.float32)
		# here we're ready for resizing in preprocess

		next_image = tf.image.resize(next_image, (image_height, image_width), antialias=True)
		next_image = next_image.numpy()

		if num_frames > 2:
			flow = demo.calculateFlow(prev_image, curr_image)
			flow2 = get_flow_vectors(prev_image, curr_image)

			h, w = flow.shape[:2]
			fx, fy = flow[:,:,0], flow[:,:,1]
			ang = np.arctan2(fy, fx) + np.pi
			v = np.sqrt(fx*fx+fy*fy)
			hsv = np.zeros((h, w, 3), np.uint8)
			hsv[...,0] = ang*(180/np.pi/2)
			hsv[...,1] = 255
			hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
			rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

			# h2, w2 = flow2.shape[:2]
			# fx2, fy2 = flow2[:,:,0], flow2[:,:,1]
			# ang2 = np.arctan2(fy2, fx2) + np.pi
			# v2 = np.sqrt(fx2*fx2+fy2*fy2)
			# hsv2 = np.zeros((h2, w2, 3), np.uint8)
			# hsv2[...,0] = ang2*(180/np.pi/2)
			# hsv2[...,1] = 255
			# hsv2[...,2] = cv2.normalize(v2, None, 0, 255, cv2.NORM_MINMAX)
			# rgb2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

			im2w = apply_optical_flow(flow, prev_image)
			mask = compute_disocclusion_mask(prev_image, curr_image, next_image)
			# im2w2 = apply_optical_flow(flow2, prev_image)
			# print(np.average(mask))
			# print(mask[0])
		
			cv2.imshow("flow",rgb)
			# cv2.imshow("flow2",rgb2)
			# cv2.imshow("1", prev_image)
			cv2.imshow("2", curr_image)

			# cv2.imshow("bool", (np.where(mask, 1, 0)).astype(np.float32))
			cv2.imshow("remapped", (im2w))
			# cv2.imshow("remapped2", (im2w2))


			cv2.waitKey(15000)
			

			cv2.destroyAllWindows()

		prev_image = curr_image
		curr_image = next_image

		# Image is now ready for line 3 of preprocess_image in stylize (resizing)
		
		# plt.imshow(image.numpy())
		# plt.show()

	return 0

# TEMPORAL STUFF

def compute_disocclusion_mask(prev_frame, curr_frame, next_frame):
	# TODO: implement weights matrix where value is 0 if pixel is disoccluded and
	# 1 otherwise?

	forward_flow = demo.calculateFlow(prev_frame, curr_frame)
	backward_flow = demo.calculateFlow(next_frame, curr_frame)

	# forward_warp = apply_optical_flow(forward_flow, prev_frame)
	# backward_warp = apply_optical_flow(backward_flow, next_frame)

	cancel_flow = forward_flow+backward_flow

	LHS = cancel_flow[:,:,0]**2 + cancel_flow[:,:,1]**2

	w_squigly_2 = forward_flow[:,:,0]**2 + forward_flow[:,:,1]**2
	w_hat_2 = backward_flow[:,:,0]**2 + backward_flow[:,:,1]**2

	RHS = .001 * (w_hat_2 + w_squigly_2) +.5

	mask = LHS <= RHS

	return mask


def get_temporal_loss(previous_stylized, previous_content, current_content, current_stylized, weights_mask):
	
	# TODO: implement temporal loss between 

	flow = demo.calculateFlow(previous_content, current_content)

	warped_style_curr = apply_optical_flow(flow, previous_stylized)

	#MAKE TF!!!
	loss = np.where(weights_mask, (current_stylized-warped_style_curr)**2, 0)
	#?????

	return np.average(loss)

def get_flow_vectors(frame_1, frame_2):



	#TODO: implement Gunner Farneback algorithm using OpenCV

	# print(frame_1.max())

	frame_1 = cv2.cvtColor(frame_1,cv2.COLOR_RGB2GRAY)
	frame_2 = cv2.cvtColor(frame_2,cv2.COLOR_RGB2GRAY)

	# print(frame_1.max())

	#Calculate Flow
	flow = cv2.calcOpticalFlowFarneback(frame_1*255,frame_2*255, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	return flow


def apply_optical_flow(flow, frame):

	# TODO: apply optical flow from frame to next frame onto stylized frame

	h, w = flow.shape[:2]
	flow = -flow
	# print(flow[:,:,0])
	# print(flow.dtype)
	flow[:,:,0] += np.arange(w)
	flow[:,:,1] += np.arange(h)[:,np.newaxis]
	flow = skimage.img_as_float32(flow)
	res = cv2.remap(frame, flow, None, cv2.INTER_LINEAR)

	return res

	# def apply_optical_flow(frame, next_frame, stylized_frame):

	# # TODO: apply optical flow from frame to next frame onto stylized frame

	# flow = get_flow_vectors(frame, next_frame)

	# h, w = flow.shape[:2]
	# flow = -flow
	# flow[:,:,0] += np.arange(w)
	# flow[:,:,1] += np.arange(h)[:,np.newaxis]

	# print(flow.shape)
	# res = cv2.remap(frame, flow, None, cv2.INTER_LINEAR)

	# return res




# These can be read in as inputs from main
video_name = "elephant.mp4"
fps = 29

video_to_images(video_name, fps)


# def video_to_images(video_name, fps):
#     video = VideoFileClip("./../data/content/video/" + video_name)

#     frames_iterable = video.iter_frames(fps=fps)
#     # This is not a normal python iterable, and can't really be used.
#     # If we want an array of all of the images, we could use this loop to get each image
#     #    and then add them one by one into an array (although this might not be necessary)

#     for frame in frames_iterable:
#         image = tf.convert_to_tensor(frame, dtype=tf.uint8)
#         image = tf.image.convert_image_dtype(image, tf.float32)

#         # Image is now ready for line 3 of preprocess_image in stylize (resizing)
		
#         # plt.imshow(image.numpy())
#         # plt.show()

# # These can be read in as inputs from main
# video_name = "tomjerry.mp4"
# fps = 1

# video_to_images(video_name, fps)

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
