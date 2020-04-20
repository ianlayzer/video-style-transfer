import numpy as np
import code.loss as loss
import code.hyperparameters as hp
from sklearn.metrics import mean_squared_error


# Layers of VGG-19 network to use for content and style feature response respectively
content_layers = ['relu4_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']


def stylize(content_image, style_image):
    """ Generates a stylized image from a content image and a stylized image

    Arguments:
        - content_image: the array representing the content image
        - style_image: the array representing the style image
    """
    # stylized image initialized as whitenoise
    stylized = np.random.normal(content_image.shape)
    # stores update gradients to update the stylized image each iteration
    delta_stylized = np.zeros(stylized.shape)
    # counter for iterations
    iteration = 0 
    # loss of previous epoch
    prev_loss = float('inf')
    # converged set to True once convergence condition satisfied
    converged = False 
    
    # perform gradient descent until loss has converged
    while not converged:
        
        # calculate loss between generated image 'stylized' and source images
        loss = loss.loss_total(content_image, style_image, stylized)
        # use backpropagation to calculate gradient of stylized image to minimize loss
        delta_stylized = loss.backprop(content_image, style_image, stylized)
        # adjust stylized image using gradient
        stylized = stylized + hp.learning_rate * delta_stylized
        
        # check if has converged
        converged = abs(loss - prev_loss) < 0.01 * loss

        print("Iteration " + str(iteration) + ": loss: " + str(loss))
        prev_loss = loss
        iteration += 1



    return stylized


# Scratchwork below... We need to get the response to the convolution layers
# for both content features and style. Content loss is the meansquared error
# between the feature map response of the content image and that of the stylized
# image

content_loss = 0
for layer in content_layers:
    target_features = np.zeros(10)
    stylized_features = np.zeros(10)
    # mean squared error between target features and stylized features
    content_loss += mean_squared_error(target_features, stylized_features)


style_loss = 0
for layer in style_layers:
    # something to do with correlation and Gram matrices
