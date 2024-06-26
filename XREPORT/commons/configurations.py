# Advanced settings for training 
#------------------------------------------------------------------------------
MIXED_PRECISION = False
USE_TENSORBOARD = False
XLA_STATE = False
ML_DEVICE = 'GPU'
NUM_PROCESSORS = 6

# Settings for training routine
#------------------------------------------------------------------------------
EPOCHS = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 10

# model settings
#------------------------------------------------------------------------------
IMG_SHAPE = (256, 256, 1)
EMBEDDING_DIMS = 512 
KERNEL_SIZE = 2
NUM_HEADS = 4
SAVE_MODEL_PLOT = True

# Settings for training data 
#------------------------------------------------------------------------------
SAMPLE_SIZE = 0.3
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.1
IMG_AUGMENT = False
MAX_CAPTION_SIZE = 200

# General settings 
#------------------------------------------------------------------------------
SEED = 54
SPLIT_SEED = 45






