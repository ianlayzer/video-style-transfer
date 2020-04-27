import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils import data_utils
import matplotlib.pyplot as plt
from model import make_vgg
import hyperparameters as hp

# refactored functions to work  with both images and video
image_height = hp.img_height
image_width = hp.img_width

content_layers = [14]
style_layers = [2, 5, 8, 13, 18]

model = make_vgg(image_height, image_width)

def preprocess_image(image_path):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
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

def stylize_img(content, style, stylized, use_temporal_loss=False, previous_stylized_frame=[]):
	# precompute content and style targets feature maps by passing through network
	# we will compare stylized responses against these at each epoch to calculate loss
	content_feature_maps = compute_all_feature_maps(content, content_layers)
	style_feature_grams = features_to_grams(compute_all_feature_maps(style, style_layers))
	# TODO: precompute optical flow for previous stylized frame?


	# optimize loss
	optimizer = tf.optimizers.Adam(learning_rate=hp.learning_rate)
	# Optimizes images to minimize loss between input content image/input style image and output stylized image
	num_epochs = hp.epoch_num
	for e in range(num_epochs):
		print("Epoch " + str(e))

		# Watches loss computation (output_stylized_img watched by default since declared as variable)
		with tf.GradientTape() as tape:
			# compute stylized features response to content and style layers
			stylized_content_features = compute_all_feature_maps(stylized, content_layers)
			stylized_style_feature_grams = features_to_grams(compute_all_feature_maps(stylized, style_layers))
			# calculate loss
			loss = get_total_loss(content_feature_maps, style_feature_grams, stylized_content_features, stylized_style_feature_grams)
			# calculate gradient of loss with respect to the stylized image (a variable)
			grad = tape.gradient(loss, stylized)
			# Applies this gradient to the image
			optimizer.apply_gradients([(grad, stylized)])
			# Clips image from 0-1, assigns gradient applied image to image variable
			stylized.assign(tf.clip_by_value(stylized, clip_value_min=0.0, clip_value_max=1.0))

	# Removes batch axis, converts image from BGR back to RGB, saves stylized image as "output.jpg" in same directory
	output_image = tf.reverse(tf.squeeze(stylized), axis=[-1]).numpy()
	tf.keras.preprocessing.image.save_img('../data/output/output.jpg', output_image)

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
	return tf.linalg.matmul(a, b)


# Gets content loss, style loss, then multiplies them by corresponding weights to get total loss
# (Weights are different than the paper, but after lots of trial and error these seem to work well)
#       They might be different due to the different optimizer?
def get_total_loss(content_features, style_feature_grams, stylized_content_features, stylized_style_feature_grams):
	content_loss = layered_mean_squared_error(content_features, stylized_content_features)
	style_loss = layered_mean_squared_error(style_feature_grams, stylized_style_feature_grams)
	return hp.content_loss_weight * content_loss + hp.style_loss_weight * style_loss

def layered_mean_squared_error(source_features, generated_features):
	total_loss = tf.constant(0.0)
	for i in range(len(source_features)):
		layer_loss = tf.keras.losses.MeanSquaredError()(source_features[i], generated_features[i])
		total_loss += layer_loss
	return total_loss



content_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('Starry_Night.jpg','https://i.ibb.co/LvGcMQd/606px-Van-Gogh-Starry-Night-Google-Art-Project.jpg')

content = preprocess_image(content_path)
style = preprocess_image(style_path)
stylized = initialize_stylized()

stylize_img(content, style, stylized)

# Uncomment this if running in Colab:
# from google.colab import files
# files.download('output.jpg')

# TODO:
# [~] fine tune stye loss weight
# [ ] try using copy of content image as inital output image instead of random noise
# [~] change learning rate in optimizer
# [ ] try other optimizer (project code uses RMSprop with learning rate = 1e-4)
# [X] adjusting image input dimensions
# [X] use other methods for computing gram matrices (eigensums)