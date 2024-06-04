# Advanced settings for training 
#------------------------------------------------------------------------------
MIXED_PRECISION = True
USE_TENSORBOARD = False
XLA_STATE = False
ML_DEVICE = 'GPU'
NUM_PROCESSORS = 6

# Settings for training routine
#------------------------------------------------------------------------------
EPOCHS = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 40

# model settings
#------------------------------------------------------------------------------
IMG_SHAPE = (256, 256, 3)
EMBEDDING_DIMS = 512 
KERNEL_SIZE = 2
NUM_HEADS = 6
SAVE_MODEL_PLOT = True

# Settings for training data 
#------------------------------------------------------------------------------
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 200
IMG_AUGMENT = False

# General settings 
#------------------------------------------------------------------------------
SEED = 54
SPLIT_SEED = 45






