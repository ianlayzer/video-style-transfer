import numpy as np
import code.loss as loss
import code.hyperparameters as hp
from sklearn.metrics import mean_squared_error
from code.nn import VGGModel

class Stylize:

    def __init__(self, content_image, style_image):
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
        self.model = VGGModel()
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

        # feature maps extracted from content image by content layers in VGG
        self.content_features = np.zeros()
        # gram matrices of feature maps extracted from style image by style layers in VGG
        self.style_features_grams = np.zeros()
        # fill in the above to feature map response matrices
        self.fill_feature_maps(self)
        # feature maps extracted from stylized (generated) image by both
        # content and style layers (calculated at each iteration)
        self.stylized_content_features = np.zeros()
        self.stylized_style_features = np.zeros()



    def fill_feature_maps(self):
        # go through content layers of network calculating response at each layer
        content_responses = []
        for layer in self.content_layers:
            # get feature response by calling specific layer o nimage
            response = self.model.__call__(layer, self.content_image)
            content_responses.append(response)
        # set field in object
        self.content_features = np.array(content_responses)

        # go through style layers of network calculating response at each layer 
        style_responses = []
        for layer in self.style_layers:
            # get feature response
            response = self.model.__call__(layer, self.style_image)
            # calculate gram matrix
            

            style_responses.append(response)
        # set field in object
        self.style_features = np.array(style_responses)



        



    # def stylize(self):
    #     # counter for iterations
    #     iteration = 0 
    #     # loss of previous epoch
    #     prev_loss = float('inf')
    #     # converged set to True once convergence condition satisfied
    #     converged = False 
        
    #     # perform gradient descent until loss has converged
    #     while not converged:
            
    #         # calculate loss between generated image 'stylized' and source images
    #         loss = loss.loss_total(content_image, style_image, stylized)
    #         # use backpropagation to calculate gradient of stylized image to minimize loss
    #         delta_stylized = loss.backprop(content_image, style_image, stylized)
    #         # adjust stylized image using gradient
    #         stylized = stylized + hp.learning_rate * delta_stylized
            
    #         # check if has converged
    #         converged = abs(loss - prev_loss) < 0.01 * loss

    #         print("Iteration " + str(iteration) + ": loss: " + str(loss))
    #         prev_loss = loss
    #         iteration += 1



    #     return stylized


# Scratchwork below... We need to get the response to the convolution layers
# for both content features and style. Content loss is the meansquared error
# between the feature map response of the content image and that of the stylized
# image

content_features = (im_height, im_weight, )
for layer in content_layers:
    content_features = np.zeros(10)
    stylized_features = np.zeros(10)
    # mean squared error between target features and stylized features
    content_loss += mean_squared_error(content_features, stylized_features)


style_loss = 0

style_style_features = np.empty(())

stylized_style_features =

for layer in style_layers:
    # something to do with correlation and Gram matrices
