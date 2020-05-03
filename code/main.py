
import os
import argparse
import tensorflow as tf
import hyperparameters as hp
from stylize import stylize_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_folder = os.path.dirname(__file__) + '../data/'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Style transfer!")
    parser.add_argument(
        '--video',
        required=False,
        action="store_true",
        help='''are you loading in a video?''')
    parser.add_argument(
        '--image',
        required=False,
        action="store_true",
        help='''are you loading in an image?''')
    parser.add_argument(
        '--content',
        required=True,
        type=str,
        help='''the content file''')   
    parser.add_argument(
        '--style',
        required=True,
        type=str,
        help='style file.')
    parser.add_argument(
        '--both',
        required=False,
        action="store_true",
        help='both short and long term consistency.')   
    parser.add_argument(
        '--short',
        required=False,
        action="store_true",
        help='enforce short term consistency.')                     

    return parser.parse_args()


def main():
    #get image paths
    content_path = data_folder + ARGS.content
    style_path = data_folder + ARGS.style
    #calling img_stylize or vid_stylize to stylize the content
    if ARGS.image:
        print(True)
        stylize_image(content_path, style_path)
    if ARGS.video:
        print(False)
        #idk what the actual video func is called
        #stylize_video(content_path, style_path)
        
    
        
    
    


#image vs. video, style, content, temporal loss (none, short, both)
#global arguments
ARGS = parse_args()


#run main
main()