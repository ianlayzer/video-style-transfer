# video-style-transfer
Style transfer on videos.

How to run:

Run main.py in terminal. Arguments are:
--image
--video
--content
--style 
--num_epochs
--learning_rate
--fps
--content_weight
--style_weight
--temporal_weight
--flow

Choose image or video, the name of the content and style files, hyperparameters, weights, and flow. Only image or video, and the content and style file names are required, as the rest have default settings. 

videotoimages.py helps vizualize optical flow and the boolean mask. It is not necessary to run the program. 

Note: to run video style transfer with flow, extra software is required (microsoft build tools and visual studio).

