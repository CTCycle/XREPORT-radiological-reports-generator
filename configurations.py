# Define general variables
#------------------------------------------------------------------------------
generate_model_graph = True
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False

# Define variables for the training
#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
epochs = 100
learning_rate = 0.001
batch_size = 100
embedding_dims = 512
kernel_size = 3
num_heads = 4

# define variables for data processing
#------------------------------------------------------------------------------
picture_size = (156, 156)
num_channels = 1
image_shape = picture_size + (num_channels,)
num_samples = 50000
test_size = 0.2
data_augmentation = False





