
import os
import argparse
import tensorflow as tf
from your_model import YourModel
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger
from keras.applications import vgg19
from keras import Model
from PIL import Image
from stylize import Stylize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_folder = os.path.dirname(__file__) + '../data/'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Style transfer!")
    parser.add_argument(
        '--video',
        required=False,
        type=str,
        help='''the video to load in''')
    parser.add_argument(
        '--image',
        required=False,
        type=str,
        help='''the video to load in''')
    parser.add_argument(
        '--content',
        required=True,
        type=str,
        help='''the video to load in''')   
    parser.add_argument(
        '--style',
        help='style file.')

    return parser.parse_args()



def main():
    content_path = data_folder + ARGS.content
    style_path = data_folder + ARGS.style
    image_input = Image.open(content_path)
    style_input = Image.open(style_path)
    w = 500
    h = 500
    model = vgg19.VGG19(input_tensor = image_input, weights = 'imagenet', include_top = False, input_shape=(h, w, 3))
    stylize = Stylize(image_input, style_input, model)
    stylize.stylize()
    #this is to get certain layers of the model
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    


#image vs. video, style, content, temporal loss (none, short, both)
#global arguments
ARGS = parse_args()


#run main
main()