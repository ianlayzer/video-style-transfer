
import os
import argparse
import tensorflow as tf
import hyperparameters as hp
from stylize import stylize_image
from stylize import stylize_video

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
data_folder = os.path.dirname(__file__) + '../data/'
framerate = 30
video_path = "./../data/content/video/elephant.mp4"
# # style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# # content_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

image_path = "./../data/content/images/Labrador.jpg"
style_path = "./../data/style/Starry_Night.jpg"




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
    #calling img_stylize or vid_stylize to stylize the content
    if ARGS.image:
        print(True)
        stylize_image(image_path, style_path)
    if ARGS.video:
        print(False)
        stylize_video(video_path, style_path, framerate)
        #idk what the actual video func is called
        #stylize_video(content_path, style_path)
        
    
        
    

#image vs. video, style, content, temporal loss (none, short, both)
#global arguments
ARGS = parse_args()


#run main
main()