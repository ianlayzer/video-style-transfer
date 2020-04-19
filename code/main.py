
import os
import argparse
import tensorflow as tf
from vgg_model import VGGModel
from your_model import YourModel
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger

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
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()



def main(args):
    video_path = data_folder + ARGS.video
    style_path = data_folder + ARGS.style



#global arguments
ARGS = parse_args():

main()