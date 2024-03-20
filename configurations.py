# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'
num_processors = 6

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 50
learning_rate = 0.0001
batch_size = 10

# Model settings
#------------------------------------------------------------------------------
picture_shape = (200, 200, 1)
embedding_dims = 512 # 768 is compatible with BioBERT embedding dimensions
kernel_size = 2
num_heads = 6
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_train_samples = 4000
num_test_samples = 400
augmentation = False

# General settings 
#------------------------------------------------------------------------------
seed = 12
split_seed = 125






