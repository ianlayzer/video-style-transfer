import tensorflow as tf
import hyperparameters as hp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc

def preprocess_image(image_path):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
	return preprocess_helper(image)

def preprocess_frame(frame):
	frame = tf.convert_to_tensor(frame, dtype=tf.uint8)
	frame = tf.image.convert_image_dtype(frame, tf.float32)
	return preprocess_helper(frame)

def preprocess_helper(image):
	image = tf.image.resize(image, (hp.img_height, hp.img_width), antialias=True)
	image = tf.image.convert_image_dtype(image, tf.uint8)
	image = tf.expand_dims(image, 0)
	image = tf.keras.applications.imagenet_utils.preprocess_input(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	return image

def preprocess_video(video_path):
	frame_list = []
	video = cv2.VideoCapture(video_path)
	i = 0
    # a variable to set how many frames you want to skip
	frame_skip = 0
	while video.isOpened():
		ret, frame = video.read()
		if not ret:
			break
		if i > frame_skip - 1:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame_list.append(preprocess_frame(frame))
			i = 0
			continue
		i += 1

	video.release()
	prCyan("- Stylizing " + str(len(frame_list)) + " frames - ")
	return frame_list

# writes a list of numpy array frames to a video
def write_video(frames, fps, filename):
	fourcc = VideoWriter_fourcc(*'mp4v')
	video = VideoWriter(filename, fourcc, fps, (hp.img_width, hp.img_height))
	for frame in frames:
		video.write(frame)
	video.release()

# makes a filename for an image
def make_filename(content_path, 
                    style_path, 
                    file_type,
                    content_loss_weight,
                    style_loss_weight,
                    temporal_loss_weight,
                    learning_rate,
                    num_epochs,
                    fps=None):
	content_name = get_filename_from_path(content_path)
	style_name = get_filename_from_path(style_path)

	name = content_name + "-" + style_name + "-" + "c_w" + str(content_loss_weight) \
		+ "-s_w" + str(style_loss_weight) + "-t_w" + str(temporal_loss_weight) \
			+ "-e" + str(num_epochs) + "-l_r" + str(learning_rate)
	# optional parameters for video
	if fps is not None:
		name += "-fps" + str(fps)

	name += file_type
	return name

# gets the name of a file from a path
def get_filename_from_path(path):
	split_path = path.split("/")
	file = split_path[len(split_path) - 1]
	split_file = file.split(".")
	return split_file[0]

def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))