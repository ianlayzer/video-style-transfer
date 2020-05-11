# weights for content and style loss (α, β, γ in Ruder et al.)
# content_loss_weight = 30000

#good numbers:   200000:0.02
content_loss_weight = 700000
style_loss_weight = 0.02
temporal_loss_weight = 4000000000
# img dimensions
img_height = 540
img_width = 960
# layer weights for style loss assuming 5 style layers
style_layer_weights = [1, 1, 1, 1, 1]

# learning rate for gradient descent
learning_rate = 0.04

# number of iterations for gradient descent
num_epochs = 700


# img_height = 224
# img_width = 224

 