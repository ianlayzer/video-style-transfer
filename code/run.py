from stylize2 import stylize_image, stylize_video
import hyperparameters as hp

video_path = "./../data/content/video/elephant_short_Trim.mp4"
# # style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# # content_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

image_path = "./../data/content/images/elephant.jpg"
style_path = "./../data/style/water-lilies.jpg"

# stylize_image(image_path, style_path, hp.content_loss_weight, hp.style_loss_weight, hp.temporal_loss_weight, hp.learning_rate, hp.num_epochs)
stylize_video(video_path, style_path, 29, hp.content_loss_weight, hp.style_loss_weight, hp.temporal_loss_weight, hp.num_epochs, hp.learning_rate, False)
