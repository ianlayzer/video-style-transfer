import tensorflow as tf
from model import make_vgg
import hyperparameters as hp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc

# refactored functions to work with both images and video
image_height = hp.img_height
image_width = hp.img_width

num_epochs = hp.num_epochs

content_layers = [14]
style_layers = [2, 5, 8, 13, 18]

style_layer_weights = hp.style_layer_weights

model = make_vgg(image_height, image_width)

def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))

def preprocess_image(image_path):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
	return preprocess_helper(image)

def preprocess_frame(frame):
	frame = tf.convert_to_tensor(frame, dtype=tf.uint8)
	frame = tf.image.convert_image_dtype(frame, tf.float32)
	return preprocess_helper(frame)

def preprocess_helper(image):
	image = tf.image.resize(image, (image_height, image_width), antialias=True)
	image = tf.image.convert_image_dtype(image, tf.uint8)
	image = tf.expand_dims(image, 0)
	image = tf.keras.applications.imagenet_utils.preprocess_input(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	return image

def initialize_stylized():
	# Output stylized image
	output_stylized_img = tf.random.normal((1, image_height, image_width, 3), mean=0.5)
	output_stylized_img = tf.clip_by_value(output_stylized_img, clip_value_min=0.0, clip_value_max=1.0)
	output_stylized_img = tf.Variable(output_stylized_img)
	return output_stylized_img

def stylize_frame(content, style, initial_stylized, precomputed_style_grams=None, use_temporal_loss=False, frames=None,  num_epochs=num_epochs):
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
	flow = []
	weights_mask = []
	if use_temporal_loss:
		weights_mask = compute_disocclusion_mask(frames[0], frames[1], frames[2])
		flow = get_flow_vectors(frames[0], frames[1])

	stylized = initial_stylized
	# we will compare stylized responses against these at each epoch to calculate loss
	content_feature_maps = compute_all_feature_maps(content, content_layers)
	style_feature_grams = precomputed_style_grams
	# check if we need to compute style target style responses now or if already computed
	if style_feature_grams is None:
		style_feature_grams = features_to_grams(compute_all_feature_maps(style, style_layers))

	# optimize loss
	optimizer = tf.optimizers.Adam(learning_rate=hp.learning_rate)
	# Optimizes images to minimize loss between input content image/input style image and output stylized image
	for e in range(num_epochs):
		# Watches loss computation (output_stylized_img watched by default since declared as variable)
		with tf.GradientTape() as tape:
			# compute stylized features response to content and style layers
			stylized_content_features = compute_all_feature_maps(stylized, content_layers)
			stylized_style_feature_grams = features_to_grams(compute_all_feature_maps(stylized, style_layers))
			# calculate loss
			content_loss, style_loss = get_total_loss(content_feature_maps, style_feature_grams, stylized_content_features, stylized_style_feature_grams, flow)
			loss = content_loss + style_loss
		if e % 100 == 0:
			print("Epoch " + str(e) + ": Content Loss = " + str(content_loss.numpy()) + " Style Loss = " + str(style_loss.numpy()))
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
	return tf.linalg.matmul(a, b) / b.size


# Gets content loss, style loss, then multiplies them by corresponding weights to get total loss
# (Weights are different than the paper, but after lots of trial and error these seem to work well)
#       They might be different due to the different optimizer?
def get_total_loss(content_features, style_feature_grams, stylized_content_features, 
					stylized_style_feature_grams, use_temporal_loss=False, previous_stylized=None,
					weights_mask=None, flow=None):
	content_loss = layered_mean_squared_error(content_features, stylized_content_features)
	style_loss = layered_mean_squared_error(style_feature_grams, stylized_style_feature_grams)

	content_loss *= hp.content_loss_weight
	style_loss *= hp.style_loss_weight
	# add temporal loss if applicable
	# if use_temporal_loss:
	# 	temporal_loss = get_temporal_loss(previous_stylized, stylized, weights_mask, flow)
	# 	total_loss += hp.temporal_loss_weight * temporal_loss
	return content_loss, style_loss

def layered_mean_squared_error(source_features, generated_features):
	total_loss = tf.constant(0.0)
	for i in range(len(source_features)):
		layer_loss = tf.keras.losses.MeanSquaredError()(source_features[i], generated_features[i])
		total_loss += layer_loss * style_layer_weights[i]
	return total_loss


# TEMPORAL STUFF

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



def get_temporal_loss(previous_stylized, current_stylized, weights_mask, flow):
	
	# TODO: implement temporal loss between 

	warped_style_curr = apply_optical_flow(flow, previous_stylized)

	loss = tf.where(weights_mask, (current_stylized-warped_style_curr)**2, 0)

	return tf.reduce_mean(loss)

def get_flow_vectors(frame_1, frame_2):

	#TODO: implement Gunner Farneback algorithm using OpenCV

	frame_1 = cv2.cvtColor(frame_1,cv2.COLOR_RGB2GRAY)
	frame_2 = cv2.cvtColor(frame_2,cv2.COLOR_RGB2GRAY)

	#Calculate Flow
	flow = cv2.calcOpticalFlowFarneback(frame_1,frame_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

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


def stylize_image(content_path, style_path):
	content = preprocess_image(content_path)
	style = preprocess_image(style_path)
	stylized = initialize_stylized()
	# stylized = tf.Variable(tf.identity(content))
	output_image = stylize_frame(content, style, stylized)

	output_image = tf.reverse(tf.squeeze(output_image), axis=[-1]).numpy()
	tf.keras.preprocessing.image.save_img('output.jpg', output_image)


def stylize_video(video_name, style_path, fps, filepath_destination):
	# get preprocessed frame list
	frame_list = preprocess_video(video_name)

	# preprocess style image
	style = preprocess_image(style_path)

	# precompute style image feature response
	style_feature_grams = features_to_grams(compute_all_feature_maps(style, style_layers))


	# starts uninitialized because there is no previous stylized frame at beginning
	previous = initialize_stylized()
	# list to add stylized frames to
	stylized_frame_list = []
	# stylize every frame
	for f in range(len(frame_list)):
		prCyan("Stylizing Frame " + str(f+1))
		# content target for this frame style transfer
		content = frame_list[f]
		# stylize img
		stylized = stylize_frame(content, style, previous, style_feature_grams)
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
		plt.imshow(output_image)
		plt.show()
		output_frames.append(output_image)
	# write video
	write_video(output_frames, fps, filepath_destination)


def preprocess_video(video_path):
	frame_list = []
	video = cv2.VideoCapture(video_path)
	i = 0
    # a variable to set how many frames you want to skip
	frame_skip = 100
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
	video = VideoWriter(filename, fourcc, fps, (image_width, image_height))
	for frame in frames:
		video.write(frame)
	video.release()

# makes a filename for an image
def make_filename(content_path, style_path, file_type, fps=None):
	content_name = get_filename_from_path(content_path)
	style_name = get_filename_from_path(style_path)

	name = content_name + "-" + style_name + "-" + "c_w" + str(hp.content_loss_weight) \
		+ "-s_w" + str(hp.style_loss_weight) + "-t_w" + str(hp.temporal_loss_weight) \
			+ "-l_r" + str(hp.learning_rate)
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



# video = "tomjerry.mp4"
# style_path = tf.keras.utils.get_file('Starry_Night.jpg','https://i.ibb.co/LvGcMQd/606px-Van-Gogh-Starry-Night-Google-Art-Project.jpg')
# # style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# # content_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

content_path = "./../data/content/images/Labrador.jpg"
style_path = "./../data/style/Starry_Night.jpg"

stylize_image(content_path, style_path)