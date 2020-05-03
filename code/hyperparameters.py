# weights for content and style loss (α, β, γ in Ruder et al.)
# content_loss_weight = 30000
content_loss_weight = 400000
style_loss_weight = .0001
temporal_loss_weight = 1
# img dimensions
img_height = 558
img_width = 700
# layer weights for style loss assuming 5 style layers
style_layer_weights = [1, 0, 0, 2, 20]

# learning rate for gradient descent
learning_rate = 0.04

# number of iterations for gradient descent
num_epochs = 600

# img_height = 224
# img_width = 224

