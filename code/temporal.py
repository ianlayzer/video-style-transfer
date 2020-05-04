import tensorflow as tf
from model import make_vgg
import hyperparameters as hp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc

def get_flow_vectors(frame_1, frame_2):

	#TODO: implement Gunner Farneback algorithm using OpenCV

	print(frame_1.shape)

	img_1 = frame_1.numpy()
	img_2 = frame_2.numpy()

	print(img_1.shape)

	img_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2GRAY)
	img_2 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY)

	#Calculate Flow
	flow = cv2.calcOpticalFlowFarneback(img_1,img_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	return flow

def apply_optical_flow(flow, stylized_frame):

	# TODO: apply optical flow from frame to next frame onto stylized frame
	img = stylized_frame.numpy()
	h, w = flow.shape[:2]
	flow = -flow
	flow[:,:,0] += np.arange(w)
	flow[:,:,1] += np.arange(h)[:,np.newaxis]
	res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)

	return tf.convert_to_tensor(res)

def compute_disocclusion_mask(prev_frame, curr_frame, next_frame):
	# TODO: implement weights matrix where value is 0 if pixel is disoccluded and
	# 1 otherwise?

	forward_flow = get_flow_vectors(prev_frame, curr_frame)
	backward_flow = get_flow_vectors(next_frame, curr_frame)

	# forward_warp = apply_optical_flow(forward_flow, prev_frame)
	# backward_warp = apply_optical_flow(backward_flow, next_frame)

	cancel_flow = forward_flow+backward_flow

	LHS = cancel_flow[:,:,0]**2 + cancel_flow[:,:,1]**2

	w_squigly_2 = forward_flow[:,:,0]**2 + forward_flow[:,:,1]**2
	w_hat_2 = backward_flow[:,:,0]**2 + backward_flow[:,:,1]**2

	RHS = .001 * (w_hat_2 + w_squigly_2) +.5

	mask = LHS <= RHS

	#Not using boolean mask rn because it is shit with Farneback optical flow
	mask.fill(1)

	mask = tf.convert_to_tensor(mask, dtype=bool)

	return mask
