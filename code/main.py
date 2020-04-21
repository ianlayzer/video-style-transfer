
import os
import argparse
import tensorflow as tf
from vgg_model import VGGModel
from your_model import YourModel
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger
from keras.applications import vgg19
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
        required=True,
        type=str,
        help='''the video to load in''')
    parser.add_argument(
        '--style',
        help='style file.')

    return parser.parse_args()



def main():
    video_path = data_folder + ARGS.video
    style_path = data_folder + ARGS.style
    image_input = Image.open(video_path)
    style_input = Image.open(style_path)
    model = vgg19.VGG19(input_tensor = image_input, weights = 'imagenet', include_top = False)
    stylize = Stylize(image_input, style_input, model)
    stylize.stylize()
    #this is to get certain layers of the model
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    

def call(dictionary, layer_name):
    return dictionary.get(layer_name)
    


#global arguments
ARGS = parse_args()


#run main
main()