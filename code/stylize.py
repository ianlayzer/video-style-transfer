import numpy as np
import hyperparameters as hp
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class Stylize:

    def __init__(self, content_image, style_image, vgg_model,
    content_weight=hp.content_weight, style_weight=hp.style_weight,
    learning_rate=hp.learning_rate, num_iterations=hp.num_iterations):
        """ Class for generating a stylized image using style transfer.

        Arguments:
            - content_image: the array representing the content image
            - style_image: the array representing the style image
        """
        # the targets for content and style respectively
        assert content_image.shape == style_image.shape
        self.content_image = content_image
        self.style_image = style_image
        # stylized image initialized as whitenoise
        self.stylized = np.random.normal(content_image.shape)
    
        # Layers of VGG-19 network to use for content and style feature response respectively
        self.model = vgg_model
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

        # feature maps extracted from content image by content layers in VGG
        self.content_features = np.zeros()
        # gram matrices of feature maps extracted from style image by style layers in VGG
        self.style_features_grams = np.zeros()
        # feature maps extracted from stylized (generated) image by both
        # content and style layers (calculated at each iteration)
        self.stylized_content_features = np.zeros()
        self.stylized_style_features_grams = np.zeros()
        # loss related
        self.content_weight = content_weight
        self.style_weight = style_weight
        # learning hyperparameters
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations


    def call(self, layer_name, input):
        layer = self.model.get_layer(layer_name)
        if layer == None:
            print("layer doesn't exist")
        else:
            layer(input)
    


    def precompute_content_features(self):
        # go through content layers of network calculating response at each layer
        content_responses = []
        for layer in self.content_layers:
            # get feature response by calling specific layer o nimage
            response = call(layer, self.content_image)
            content_responses.append(response)
        # set field in object
        self.content_features = np.array(content_responses)

    def precopute_style_features(self):
        # go through style layers of network calculating response at each layer 
        style_responses = []
        for layer in self.style_layers:
            # get feature response
            response = call(layer, self.style_image)
            # calculate gram matrix
            gram = (response.T @ response)
            style_responses.append(response)
        # set field in object
        self.style_features = np.array(style_responses)


    def compute_stylized_features(self):
        return 0


    def stylize(self):
        # precompute feature maps for content and style targets (only done once)
        self.precompute_content_features()
        self.precopute_style_features()
        self.compute_stylized_features()
        
        weighted_content_loss = 0
        weighted_style_loss = 0
        
        
        stylized = tf.Variables(self.stylized)

        # calculate content loss
        content_loss = tf.losses.mean_squared_error(self.content_features, self.stylized_content_features)
        content_loss = tf.multiply(self.content_weight, content_loss)
        # calculate style loss
        style_loss = tf.mean_squared_error()
        style_loss = tf.multiply(self.style_weight, style_loss)
        loss = tf.add(content_loss, style_loss)
        
        
        # use gradient descent optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        for i in range(0, self.num_iterations):


            print("Iteration " + str(i))



        
        return self.stylized
    


