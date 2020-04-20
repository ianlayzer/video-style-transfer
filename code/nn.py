from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import SGD
import cv2, numpy as np

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)


        self.vgg19 = [

        Conv2D(64, 3, 3, activation='relu'),
        Conv2D(64, 3, 3, activation='relu'),
        MaxPool2D((2,2), strides=(2,2)),

        Conv2D(128, 3, 3, activation='relu'),
        Conv2D(128, 3, 3, activation='relu'),
        MaxPool2D((2,2), strides=(2,2)),

        Conv2D(256, 3, 3, activation='relu'),
        Conv2D(256, 3, 3, activation='relu'),
        Conv2D(256, 3, 3, activation='relu'),
        Conv2D(256, 3, 3, activation='relu'),
        MaxPool2D((2,2), strides=(2,2)),

        Conv2D(512, 3, 3, activation='relu'),
        Conv2D(512, 3, 3, activation='relu'),
        Conv2D(512, 3, 3, activation='relu'),
        Conv2D(512, 3, 3, activation='relu'),
        MaxPool2D((2,2), strides=(2,2)),

        Conv2D(512, 3, 3, activation='relu'),
        Conv2D(512, 3, 3, activation='relu'),
        Conv2D(512, 3, 3, activation='relu'),
        Conv2D(512, 3, 3, activation='relu'),
        MaxPool2D((2,2), strides=(2,2)),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax'),
        ]
        
        for l in self.vgg19:
            l.trainable = False

    """ Passes the image through the network. """
   def call(self, img):
        for layer in self.vgg19:
            img = layer(img)
        return img





if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_19('vgg19_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print np.argmax(out)