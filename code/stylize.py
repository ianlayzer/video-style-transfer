import tensorflow as tf
from model import make_vgg
import hyperparameters as hp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
from processing import *
from temporal import *

# refactored functions to work with both images and video
image_height = hp.img_height
image_width = hp.img_width

content_layers = [14]
style_layers = [2, 5, 8, 13, 18]

style_layer_weights = hp.style_layer_weights

model = make_vgg(image_height, image_width)


def stylize_image(content_path, 
					style_path, 
					content_loss_weight,
					style_loss_weight,
					temporal_loss_weight,
					learning_rate,
					num_epochs):
	content = preprocess_image(content_path)
	style = preprocess_image(style_path)
	stylized = content
	# initialize_stylized()
	# stylized = tf.Variable(tf.identity(content))
	output_image = stylize_frame(curr_content =content, 
								prev_content = content,
								prev_prev_content= content,
								style=style, 
								initial_stylized=stylized, 
								content_loss_weight=content_loss_weight,
								style_loss_weight=style_loss_weight,
								temporal_loss_weight=temporal_loss_weight,
								num_epochs=num_epochs,
								learning_rate=learning_rate,
								use_temporal_loss=False)

	output_image = tf.reverse(tf.squeeze(output_image), axis=[-1]).numpy()

	name = "./../data/output/images/"
	name += make_filename(content_path=content_path, 
							style_path=style_path, 
							file_type=".jpg",
							content_loss_weight=content_loss_weight,
							style_loss_weight=style_loss_weight,
							temporal_loss_weight=temporal_loss_weight,
							num_epochs=num_epochs,
							learning_rate=learning_rate)
	tf.keras.preprocessing.image.save_img(name, output_image)

def initialize_stylized():
	# Output stylized image
	output_stylized_img = tf.random.normal((1, image_height, image_width, 3), mean=0.5)
	output_stylized_img = tf.clip_by_value(output_stylized_img, clip_value_min=0.0, clip_value_max=1.0)
	output_stylized_img = tf.Variable(output_stylized_img)
	return output_stylized_img	


def stylize_video(video_path, 
					style_path, 
					fps, 
					content_loss_weight,
					style_loss_weight,
					temporal_loss_weight,
					num_epochs,
					learning_rate,
					use_temporal_loss):
	# get preprocessed frame list
	frame_list = preprocess_video(video_path)

	# preprocess style image
	style = preprocess_image(style_path)

	# precompute style image feature response
	style_feature_grams = features_to_grams(compute_all_feature_maps(style, style_layers))


	# starts uninitialized because there is no previous stylized frame at beginning
	previous = frame_list[0]
	# initialize_stylized()
	# list to add stylized frames to
	stylized_frame_list = []
	# stylize every frame
	for f in range(len(frame_list)):
		prCyan("Stylizing Frame " + str(f+1))
		# content target for this frame style transfer
		curr_content = frame_list[f]
		# stylize img
		if f < 2:
			stylized = stylize_frame(curr_content=curr_content,
									prev_content=curr_content,
									prev_prev_content=curr_content, 
									style=style, 
									initial_stylized=previous, 
									precomputed_style_grams=style_feature_grams,
									use_temporal_loss=False,
									content_loss_weight=content_loss_weight,
									style_loss_weight=style_loss_weight,
									temporal_loss_weight=temporal_loss_weight,
									learning_rate=learning_rate,
									num_epochs=num_epochs)
		else:
			prev_content = frame_list[f-1]
			prev_prev_content = frame_list[f-2]
			stylized = stylize_frame(curr_content=curr_content,
									prev_content=prev_content,
									prev_prev_content=prev_prev_content, 
									style=style, 
									initial_stylized=previous, 
									precomputed_style_grams=style_feature_grams,
									use_temporal_loss=use_temporal_loss,
									content_loss_weight=content_loss_weight,
									style_loss_weight=style_loss_weight,
									temporal_loss_weight=temporal_loss_weight,
									learning_rate=learning_rate,
									num_epochs=num_epochs)
		# add to stylized frame list
		to_append = tf.identity(stylized)
		stylized_frame_list.append(to_append)

		# update previous stylized frame to the frame we just stylized with optical flow applied
		previous = stylized
		# TODO: MAKE THIS WORK f, f+1, just numbers
		# initial_stylized = apply_optical_flow(f, f+1, stylized)
	output_frames = []
	for stylized_image in stylized_frame_list:
		output_image = tf.squeeze(stylized_image).numpy()
		output_image = cv2.normalize(output_image, None, 0 , 255,cv2.NORM_MINMAX,cv2.CV_8U)
		# plt.imshow(output_image)
		# plt.show()
		output_frames.append(output_image)
	# write video
	output_filepath = "./../data/output/video/"
	output_filepath += make_filename(content_path=video_path, 
										style_path=style_path, 
										file_type=".mp4", 
										fps=fps,
										content_loss_weight=content_loss_weight,
										style_loss_weight=style_loss_weight,
										temporal_loss_weight=temporal_loss_weight,
										learning_rate=learning_rate,
										num_epochs=num_epochs)
	write_video(output_frames, fps, output_filepath)

def stylize_frame(curr_content,
					prev_content,
					prev_prev_content,
					style, 
					initial_stylized,  
					content_loss_weight,
					style_loss_weight,
					temporal_loss_weight,
					learning_rate,
					num_epochs, 
					use_temporal_loss,
					precomputed_style_grams=None):
	"""Generates a stylized still image frame using the content from content, the
	style from style. The stylized image is initialized as the inputted stylized image.
	We can also pass in stylized feature maps rather than a stylized image, in which
	case we do not need to recompute the feature maps. We include temporal loss in
	total loss if use_temporal_loss is True.

	Arguments:
		- content: the content target image, already processed (tensorflow variable)
		- style: the style target image, already processed (tensorflow variable)
		- initial_stylized: the initialized value of our stylized image, we will optimize
					starting from this value. If stylizing an image, we pass in
					whitenoise. If stylizing a frame of a video, we pass in
					whitenoise for the first frame, and the previous stylized frame
					for every subsequent frame.
		- precomputed_style_grams: when stylizing a video, we do not want to recompute the
					style feature maps for the style target in every frame.
					instead, we should compute once and then pass in the
					feature map gram matrices to this function for every frame
		- use_temporal_loss: whether or not to include temporal loss in the total
					loss calculation
		- frames: a list [prev_frame, curr_frame, next_frame]
	"""
	# the previous stylized frame
	# previous_stylized = tf.identity(initial_stylized)
	# TODO: temporal weights mask
	print(use_temporal_loss, "use_loss")
	flow = []
	weights_mask = []
	stylized = tf.Variable(initial_stylized)
	if use_temporal_loss:
		weights_mask = compute_disocclusion_mask(prev_prev_content, prev_content, curr_content)
		flow = get_flow_vectors(prev_content, curr_content)
		stylized = tf.Variable(apply_optical_flow(flow, initial_stylized))

	# we will compare stylized responses against these at each epoch to calculate loss
	content_feature_maps = compute_all_feature_maps(curr_content, content_layers)
	style_feature_grams = precomputed_style_grams
	# check if we need to compute style target style responses now or if already computed
	if style_feature_grams is None:
		style_feature_grams = features_to_grams(compute_all_feature_maps(style, style_layers))
	# optimize loss
	optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
	# Optimizes images to minimize loss between input content image/input style image and output stylized image
	for e in range(num_epochs):
		# Watches loss computation (output_stylized_img watched by default since declared as variable)
		with tf.GradientTape() as tape:
			# compute stylized features response to content and style layers
			stylized_content_features = compute_all_feature_maps(stylized, content_layers)
			stylized_style_feature_grams = features_to_grams(compute_all_feature_maps(stylized, style_layers))
			# calculate loss
			content_loss = content_loss_weight * layered_mean_squared_error(content_feature_maps, stylized_content_features)
			style_loss = style_loss_weight * layered_mean_squared_error(style_feature_grams, stylized_style_feature_grams)
			temporal_loss = tf.constant(0.0)
			if use_temporal_loss:
				temporal_loss = temporal_loss_weight * get_temporal_loss(initial_stylized, stylized, weights_mask, flow)
			loss = content_loss + style_loss + temporal_loss
			# add temporal loss if applicable
			# if use_temporal_loss:
				# TODO: temporal loss

		if e % 100 == 0:
			print("Epoch " + str(e) + ": Content Loss = " + str(content_loss.numpy()) + " Style Loss = " + str(style_loss.numpy()), " Temporal Loss = " + str(temporal_loss.numpy()))
		# calculate gradient of loss with respect to the stylized image (a variable)
		grad = tape.gradient(loss, stylized)
		# Applies this gradient to the image
		optimizer.apply_gradients([(grad, stylized)])
		# Clips image from 0-1, assigns gradient applied image to image variable
		stylized.assign(tf.clip_by_value(stylized, clip_value_min=0.0, clip_value_max=1.0))
		
		# if e % 100 == 0 and e != 0:
		# 	output_image = tf.reverse(tf.squeeze(stylized), axis=[-1]).numpy()
		# 	tf.keras.preprocessing.image.save_img('epoch' + str(e) + '.jpg', output_image)

	# return to be used as initial stylized for next frame
	return stylized

def initialize_stylized():
	# Output stylized image
	output_stylized_img = tf.random.normal((1, image_height, image_width, 3), mean=0.5)
	output_stylized_img = tf.clip_by_value(output_stylized_img, clip_value_min=0.0, clip_value_max=1.0)
	output_stylized_img = tf.Variable(output_stylized_img)
	return output_stylized_img

# computes list of feature map responses by passing image through network
# up until each layer in layers
def compute_all_feature_maps(image, layers):
	maps = []
	for layer in layers:
		feat = compute_feature_map(image, layer)
		maps.append(feat)
	return maps

# Feeds image through portion of VGG (depending on content or style model)
# Returns feature map for that image
def compute_feature_map(img, max_layer):
	img_copy = img
	for l in range(max_layer):
		curr_layer = model.get_layer(index=l)
		img_copy = curr_layer(img_copy)
	return img_copy

def features_to_grams(feature_maps):
	grams = []
	for i in range(len(feature_maps)):
		g = compute_feature_map_gram(feature_maps[i])
		grams.append(g)
	return grams
	
# Vectorizes feature map, then computes its Gram matrix
def compute_feature_map_gram(feature_map):
	depth = feature_map.shape[3]
	b = tf.reshape(tf.squeeze(feature_map) , [-1, depth])
	a = tf.transpose(b)
	
	# a and b have the same dimensions
	# both are 2 dimensional, vector length x depth where vector length = width x height of each feature map
	# since depth is constant regardless of the input image size, we don't really need this value (unless you want it)
	# so size =

	return tf.linalg.matmul(a, b)

def layered_mean_squared_error(source_features, generated_features):
	total_loss = tf.constant(0.0)
	for i in range(len(source_features)):
		layer_loss = tf.keras.losses.MeanSquaredError()(source_features[i], generated_features[i])
		total_loss += layer_loss * style_layer_weights[i]
	return total_loss

def get_temporal_loss(previous_stylized, current_stylized, weights_mask, flow):
	
	# TODO: implement temporal loss between 

	# print(flow)
	warped_style_curr = apply_optical_flow(flow, previous_stylized)
	# print(warped_style_curr-current_stylized, "warped-stylized")
	# print(current_stylized, "curr_stylized")

	loss = tf.where(weights_mask, (current_stylized-warped_style_curr)**2, 0)

	return tf.reduce_mean(loss)