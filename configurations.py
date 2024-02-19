# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'
num_processors = 6

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 1
learning_rate = 0.001
batch_size = 25

# Model settings
#------------------------------------------------------------------------------
picture_shape = (256, 256, 1)
embedding_dims = 512
kernel_size = 2
num_heads = 4
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_train_samples = 500
num_test_samples = 100
augmentation = False

# General settings 
#------------------------------------------------------------------------------
seed = 72






