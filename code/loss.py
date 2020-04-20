import numpy as np
from sklearn.metrics import mean_squared_error
import code.hyperparameters as hp



def loss_content(content_image, stylized_image):

    return 0.0

def loss_style(style_image, stylized_image):
    return 0.0

def loss_total(content_image, style_image, stylized_image):
    """ Calculates the total loss for a stylized image based on content loss
    and style loss. We minimize this loss function through gradient descent
    to produce a stylized image.

    Arguments:
        content_image - a (im_height * im_width * 3) numpy array representing
                        the image to be used as the basis of the CONTENT
                        of the generated stylized image.
        style_image - a (im_height * im_width * 3) numpy array representing
                        the image to be used as the basis of the STYLE
                        of the generated stylized image.
        stylized_image - a (im_height * im_width * 3) numpy array representing
                        the STYLIZED image that is being produced. At the
                        beginning of the generation process, this is initialized
                        as whitenoise.
    Returns:
        The float loss between the stylized image and the content/style images. 
    """
    return hp.alpha * loss_content(content_image, stylized_image) + \
    hp.beta * loss_style(style_image, stylized_image)


def backprop_content(content_image, stylized):
    
    return np.zeros(stylized.shape)


def backprop_style(style_image, stylized):

    return np.zeros(stylized.shape)

def backprop_total(content_image, style_image, stylized):
    """ Uses standard error backpropagation to get the gradient of the stylized
    image matrix in the direction that minimzes the loss function.
    
    Arguments:
        content_image - a (im_height * im_width * 3) numpy array representing
                        the image to be used as the basis of the CONTENT
                        of the generated stylized image.
        style_image - a (im_height * im_width * 3) numpy array representing
                        the image to be used as the basis of the STYLE
                        of the generated stylized image.
        stylized_image - a (im_height * im_width * 3) numpy array representing
                        the STYLIZED image that is being produced. At the
                        beginning of the generation process, this is initialized
                        as whitenoise.
    Returns:
        A (im_height, im_width, 3) numpy array representing the gradient of the
        stylized image in the direction that minimzes the loss function.
    """

    return hp.alpha * backprop_content(content_image, stylized) + \
    hp.beta * backprop_style(style_image, stylized)