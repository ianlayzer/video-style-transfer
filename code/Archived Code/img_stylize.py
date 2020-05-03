import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils import data_utils
from model import make_vgg
import hyperparameters as hp

# (larger dimensions produces clearer final images but takes longer to run)

image_height = hp.img_height
image_width = hp.img_width
max_layers = [14, 2, 5, 8, 13, 18]
model = make_vgg(image_height, image_width)
  
def stylize_image(content_file, style_file):
### VGG 19, max pooling layers replaced with average pooling, input layer changed for variable sizes ###
  model.summary()
  for layer in model.layers: # necessary?
    layer.trainable = False

  ##############################
  content_path = content_file
  style_path = style_file
  # URL to content image (easy to replace with reading from file later)
  #content_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    # content_path = tf.keras.utils.get_file('MonaLisa.jpg', 'https://i.ibb.co/Mk3SjBT/monalisa.jpg')
  # URL to style image
    # style_path = tf.keras.utils.get_file('Kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
  #style_path = tf.keras.utils.get_file('Starry_Night.jpg','https://i.ibb.co/LvGcMQd/606px-Van-Gogh-Starry-Night-Google-Art-Project.jpg')
  # three images
  input_content_img = preprocess_image(content_path)
  input_style_img = preprocess_image(style_path)
  output_stylized_img = initialize_stylized()

  ### CONTENT LOSS ###
  # Maximum VGG layers for content and style models
  # max layers = [content, style1, style2, style3, style4, style5]
  
  # The original paper used L-BFGS as their optimizer, but it's not available in tensorflow
  # Used Adam instead, tuned learning rate (feel free to adjust it and compare results--0.03/0.04 worked well)
  optimizer = tf.optimizers.Adam(learning_rate=hp.learning_rate)
  # Optimizes images to minimize loss between input content image/input style image and output stylized image
  num_epochs = hp.epoch_num
  for e in range(num_epochs):
    # Prints epoch number every 100 epochs
    if e % 100 == 0:
      print("Epoch " + str(e))
    # Watches loss computation (output_stylized_img watched by default since declared as variable)
    with tf.GradientTape() as tape:
      loss = get_total_loss(input_style_img, output_stylized_img, input_content_img)
    # Computes gradient between the loss and the output stylized image
    grad = tape.gradient(loss, output_stylized_img)
    # Applies this gradient to the image
    optimizer.apply_gradients([(grad, output_stylized_img)]) 
    # Clips image from 0-1, assigns gradient applied image to image variable
    output_stylized_img.assign(tf.clip_by_value(output_stylized_img, clip_value_min=0.0, clip_value_max=1.0))


# Removes batch axis, converts image from BGR back to RGB, saves stylized image as "output.jpg" in same directory
  output_image = tf.reverse(tf.squeeze(output_stylized_img), axis=[-1]).numpy()
  tf.keras.preprocessing.image.save_img('output.jpg', output_image)


# Reads in and decodes images
# Converts images to correct datatype for resizing (float 0-1)
# Resizes images to desired dimensions
# Converts images back uint 0-255 for preprocessing
# Adds batch axis for VGG
# Converts images from RGB to BGR, standardizes them based on mean and stdd of imagenet
# Converts images back to float so that gradients can be applied during optimization
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
def get_content_loss(input_content, output_stylized):
  return tf.keras.losses.MeanSquaredError()(get_feature_map(input_content, 0), get_feature_map(output_stylized, 0))

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
def get_style_loss(style_input_img, style_output_img):
  total_style_loss = tf.constant(0.0)
  for i in range(1, 6):
    feature_map_in = get_feature_map(style_input_img, i)
    feature_map_out = get_feature_map(style_output_img, i)

    depth = feature_map_in.shape[3]
    gram_in_style = compute_feature_map_gram(feature_map_in, depth)
    gram_out_stylized = compute_feature_map_gram(feature_map_out, depth)

    style_loss = tf.keras.losses.MeanSquaredError()(gram_in_style, gram_out_stylized)
    total_style_loss += style_loss

  return total_style_loss

# Gets content loss, style loss, then multiplies them by corresponding weights to get total loss
# (Weights are different than the paper, but after lots of trial and error these seem to work well)
#       They might be different due to the different optimizer?
def get_total_loss(input_style_img, output_stylized_img, input_content_img):
  content_loss_weight = hp.content_loss_weight
  style_loss_weight = hp.style_loss_weight # increasing to 0.05 also worked well
  content_loss = get_content_loss(input_content_img, output_stylized_img)
  style_loss = get_style_loss(input_style_img, output_stylized_img)

  total_loss = content_loss_weight*content_loss + style_loss_weight*style_loss
  return total_loss



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