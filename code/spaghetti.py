import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils import data_utils
import matplotlib.pyplot as plt
import PIL.Image

# (larger dimensions produces clearer final images but takes longer to run)
img_height = 224
img_width = 224

### VGG 19, max pooling layers replaced with average pooling, input layer changed for variable sizes ###

img_input = layers.Input(shape=(img_height, img_width, 3))
# Block 1
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

model = tf.keras.Model(img_input, x)

# Load in weights
weights_no_top = ('https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
weights_path = data_utils.get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', weights_no_top)
model.load_weights(weights_path)

# model.summary()

for layer in model.layers: # necessary?
  layer.trainable = False

##############################

# URL to content image (easy to replace with reading from file later)
content_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    # content_path = tf.keras.utils.get_file('MonaLisa.jpg', 'https://i.ibb.co/Mk3SjBT/monalisa.jpg')
# URL to style image
    # style_path = tf.keras.utils.get_file('Kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
style_path = tf.keras.utils.get_file('Starry_Night.jpg','https://i.ibb.co/LvGcMQd/606px-Van-Gogh-Starry-Night-Google-Art-Project.jpg')


# Reads in and decodes images
# Converts images to correct datatype for resizing (float 0-1)
# Resizes images to desired dimensions
# Converts images back uint 0-255 for preprocessing
# Adds batch axis for VGG
# Converts images from RGB to BGR, standardizes them based on mean and stdd of imagenet
# Converts images back to float so that gradients can be applied during optimization
def preprocess_images(content_path, style_path):
  # Input content image
  input_content_img = tf.io.read_file(content_path)
  input_content_img = tf.image.decode_image(input_content_img, channels=3, dtype=tf.float32)
  input_content_img = tf.image.resize(input_content_img, (img_height, img_width)) # add antialiasing argument
  input_content_img = tf.image.convert_image_dtype(input_content_img, tf.uint8)
  input_content_img = tf.expand_dims(input_content_img, 0)
  input_content_img = tf.keras.applications.imagenet_utils.preprocess_input(input_content_img)
  input_content_img = tf.image.convert_image_dtype(input_content_img, tf.float32)

  # Input style image
  input_style_img = tf.io.read_file(style_path)
  input_style_img = tf.image.decode_image(input_style_img, channels=3, dtype=tf.float32)
  input_style_img = tf.image.resize(input_style_img, (img_height, img_width)) # add antialiasing argument
  input_style_img = tf.image.convert_image_dtype(input_style_img, tf.uint8)
  input_style_img = tf.expand_dims(input_style_img, 0)
  input_style_img = tf.keras.applications.imagenet_utils.preprocess_input(input_style_img)
  input_style_img = tf.image.convert_image_dtype(input_style_img, tf.float32)

  # Output stylized image
  output_stylized_img = tf.random.normal((1, img_height, img_width, 3), mean=0.5)
  output_stylized_img = tf.clip_by_value(output_stylized_img, clip_value_min=0.0, clip_value_max=1.0)
  output_stylized_img = tf.Variable(output_stylized_img)

  return input_content_img, input_style_img, output_stylized_img

input_content_img, input_style_img, output_stylized_img = preprocess_images(content_path, style_path)

### CONTENT LOSS ###

# Maximum VGG layers for content and style models
# max layers = [content, style1, style2, style3, style4, style5]
max_layers = [14, 2, 5, 8, 13, 18]

# Feeds image through portion of VGG (depending on content or style model)
# Returns feature map for that image
def get_feature_map(img, max_layers_index):
  img_copy = img
  max_layer = max_layers[max_layers_index]
  for l in range(max_layer):
    curr_layer = model.get_layer(index=l)
    img_copy = curr_layer(img_copy)

  return img_copy

# Computes content loss as MSE between the feature maps of the input content image and the output stylized image
def get_content_loss():
  return tf.keras.losses.MeanSquaredError()(get_feature_map(input_content_img, 0), get_feature_map(output_stylized_img, 0))

### STYLE LOSS ###

# Vectorizes feature map, then computes its Gram matrix
def compute_feature_map_gram(feature_map, depth):
  b = tf.reshape(tf.squeeze(feature_map) , [-1, depth])
  a = tf.transpose(b)
  return tf.linalg.matmul(a, b)

# Feeds input style image, output stylized image through each of the style models (subsections of VGG)
# Computes the Gram matrix for each image's resulting feature map
# Computes loss as MSE between Gram matrices of the images' feture maps
# Sums these losses for each style model to get total style loss
def get_style_loss():
  total_style_loss = tf.constant(0.0)
  for i in range(1, 6):
    feature_map_in = get_feature_map(input_style_img, i)
    feature_map_out = get_feature_map(output_stylized_img, i)

    depth = feature_map_in.shape[3]
    gram_in_style = compute_feature_map_gram(feature_map_in, depth)
    gram_out_stylized = compute_feature_map_gram(feature_map_out, depth)

    style_loss = tf.keras.losses.MeanSquaredError()(gram_in_style, gram_out_stylized)
    total_style_loss += style_loss

  return total_style_loss

# Gets content loss, style loss, then multiplies them by corresponding weights to get total loss
# (Weights are different than the paper, but after lots of trial and error these seem to work well)
#       They might be different due to the different optimizer?
def get_total_loss():
  content_loss_weight = 10000
  style_loss_weight = 0.03
  content_loss = get_content_loss()
  style_loss = get_style_loss()

  total_loss = content_loss_weight*content_loss + style_loss_weight*style_loss
  return total_loss

# The original paper used L-BFGS as their optimizer, but it's not available in tensorflow
# Used Adam instead, tuned learning rate (feel free to adjust it and compare results--0.03/0.04 worked well)
optimizer = tf.optimizers.Adam(learning_rate=0.04)

# Optimizes images to minimize loss between input content image/input style image and output stylized image
num_epochs = 1000
for e in range(num_epochs):
  # Prints epoch number every 100 epochs
  if e % 100 == 0:
    print("Epoch " + str(e))
  # Watches loss computation (output_stylized_img watched by default since declared as variable)
  with tf.GradientTape() as tape:
    loss = get_total_loss()
  # Computes gradient between the loss and the output stylized image
  grad = tape.gradient(loss, output_stylized_img)
  # Applies this gradient to the image
  optimizer.apply_gradients([(grad, output_stylized_img)]) 
  # Clips image from 0-1, assigns gradient applied image to image variable
  output_stylized_img.assign(tf.clip_by_value(output_stylized_img, clip_value_min=0.0, clip_value_max=1.0))


# Removes batch axis, converts image from BGR back to RGB
    # If running in Colab (will take a very long time):
# plt.imshow(tf.reverse(tf.squeeze(output_stylized_img), axis=[-1]).numpy())

    # If running in GCP (saves stylized image as "output.jpg" in same directory):
output_image = tf.reverse(tf.squeeze(output_stylized_img), axis=[-1]).numpy()
tf.keras.preprocessing.image.save_img('output.jpg', output_image)

# TODO:
# [~] fine tune stye loss weight
# [ ] try using copy of content image as inital output image instead of random noise
# [~] change learning rate in optimizer
# [ ] try other optimizer (project code uses RMSprop with learning rate = 1e-4)
# [X] rewrite code to feed through layers with for loops instead of creating separate models
# [X] try adjusting image input dimensions
# [X] try using other methods for computing gram matrices (eigensums)