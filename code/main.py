
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
from img_stylize import stylize_image

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
        help='''are you loading in a video?''')
    parser.add_argument(
        '--image',
        required=False,
        type=str,
        help='''are you loading in an image?''')
    parser.add_argument(
        '--content',
        required=True,
        type=str,
        help='''the content file''')   
    parser.add_argument(
        '--style',
        required=True
        help='style file.')
    parser.add_argument(
        '--both',
        required=False
        help='both short and long term consistency.')   
    parser.add_argument(
        '--short',
        required=False
        help='enforce short term consistency.')                     

    return parser.parse_args()



def main():
    #get image paths
    content_path = data_folder + ARGS.content
    style_path = data_folder + ARGS.style
    #calling img_stylize or vid_stylize to stylize the content
    if ARGS.image:
        stylize_image(image_input, style_input)
    
        
    
    


#image vs. video, style, content, temporal loss (none, short, both)
#global arguments
ARGS = parse_args()


#run main
main()