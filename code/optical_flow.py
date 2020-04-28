
import cv2
import tensorflow as tf
import numpy as np

image_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
image = tf.image.convert_image_dtype(image, tf.uint8)
image = tf.expand_dims(image, 0)
image = tf.keras.applications.imagenet_utils.preprocess_input(image)
image = tf.image.convert_image_dtype(image, tf.float32)



def get_optical_flow(img_1, img_2):

    "finds optical flow from one image to the next and return those flow vectors"

    # Convert to Grayscale

    print(img_1.numpy().shape)

    img_1 = np.reshape(img_1.numpy(), (577, 700, 3))
    img_2 = np.reshape(img_2.numpy(), (577, 700, 3))


    img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)


    #Calculate Flow
    flow = cv2.calcOpticalFlowFarneback(img_1,img_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

print(get_optical_flow(image, image))
