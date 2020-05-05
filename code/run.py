from stylize import stylize_image, stylize_video
import hyperparameters as hp

video_path = "./../data/content/video/tom_jerry_short.mp4"
# # style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# # content_path = tf.keras.utils.get_file('Labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

content_path = "./../data/content/images/Labrador.jpg"
style_path = "./../data/style/Starry_Night.jpg"

<<<<<<< HEAD
stylize_video(video_path, style_path, 1, hp.content_loss_weight, hp.style_loss_weight, hp.temporal_loss_weight, hp.num_epochs, hp.learning_rate, False)
=======
stylize_image(content_path, style_path, num_epochs=1)
>>>>>>> master
