# Define general variables
#------------------------------------------------------------------------------
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

# Define variables for the training
#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
epochs = 1
learning_rate = 0.001
batch_size = 50
embedding_dims = 512
kernel_size = 2
num_heads = 4

# define variables for data processing
#------------------------------------------------------------------------------
picture_shape = (144, 144, 1)
num_train_samples = 5000
num_test_samples = 500
augmentation = False





