from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import SGD
import cv2, numpy as np


def call ()




# class VGGModel(tf.keras.Model):
#     def __init__(self):
#         super(VGGModel, self).__init__()

#         # Optimizer
#         self.optimizer = tf.keras.optimizers.RMSprop(
#             learning_rate=hp.learning_rate,
#             momentum=hp.momentum)


#         self.vgg19 = [

#         Conv2D(64, 3, 3, activation='relu', name="relu1_1"),
#         Conv2D(64, 3, 3, activation='relu', name="relu1_2"),
#         MaxPool2D((2,2), strides=(2,2), name="pool1"),

#         Conv2D(128, 3, 3, activation='relu', name="relu2_1"),
#         Conv2D(128, 3, 3, activation='relu', name="relu2_2"),
#         MaxPool2D((2,2), strides=(2,2), name="pool2"),

#         Conv2D(256, 3, 3, activation='relu', name="relu3_1"),
#         Conv2D(256, 3, 3, activation='relu', name="relu3_2"),
#         Conv2D(256, 3, 3, activation='relu', name="relu3_3"),
#         Conv2D(256, 3, 3, activation='relu', name="relu3_4"),
#         MaxPool2D((2,2), strides=(2,2), name="pool3"),

#         Conv2D(512, 3, 3, activation='relu', name="relu4_1"),
#         Conv2D(512, 3, 3, activation='relu', name="relu4_2"),
#         Conv2D(512, 3, 3, activation='relu', name="relu4_3"),
#         Conv2D(512, 3, 3, activation='relu', name="relu4_4"),
#         MaxPool2D((2,2), strides=(2,2), name="pool4"),

#         Conv2D(512, 3, 3, activation='relu', name="relu5_1"),
#         Conv2D(512, 3, 3, activation='relu', name="relu5_2"),
#         Conv2D(512, 3, 3, activation='relu', name="relu5_3"),
#         Conv2D(512, 3, 3, activation='relu', name="relu5_4"),
#         MaxPool2D((2,2), strides=(2,2), name="pool5"),

#         Flatten(),
#         Dense(4096, activation='relu', name="relu6"),
#         Dropout(0.5, name="drop6"),
#         Dense(4096, activation='relu', name="relu7"),
#         Dropout(0.5, name="drop7"),
#         Dense(10, activation='softmax', name="prob"),
#         ]
        
#         for l in self.vgg19:
#             l.trainable = False

#     """ Passes the image through the network. """
#     def call(self, img, layer_name):
#         for layer in self.vgg19:
#             if (layer.name == layer_name):
#                 break
#             else:
#                 img = layer(image)
#         return img
