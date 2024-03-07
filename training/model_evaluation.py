import os
import sys
import pandas as pd
import tensorflow as tf

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

# import modules and classes
#------------------------------------------------------------------------------    
from utils.data_assets import PreProcessing, DataGenerator, TensorDataSet
from utils.model_assets import ModelTraining, Inference
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
images_path = os.path.join(globpt.data_path, 'images') 
cp_path = os.path.join(globpt.train_path, 'checkpoints') 
os.mkdir(images_path) if not os.path.exists(images_path) else None
os.mkdir(cp_path) if not os.path.exists(cp_path) else None
 

# [LOAD MODEL AND DATA]
#==============================================================================
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
XREPORT model evaluation
-------------------------------------------------------------------------------
...
''') 

# Load pretrained model and its parameters
#------------------------------------------------------------------------------
inference = Inference(cnf.seed) 
model, parameters = inference.load_pretrained_model(cp_path)
model_path = inference.folder_path
model.summary()

# Load the tokenizer
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
preprocessing_path = os.path.join(model_path, 'preprocessing')
tokenizer = preprocessor.load_tokenizer(preprocessing_path, 'word_tokenizer')
vocab_size = len(tokenizer.word_index) + 1

# load preprocessed csv files (train and test datasets)
#------------------------------------------------------------------------------
file_loc = os.path.join(preprocessing_path, 'XREP_train.csv') 
train_data = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)
file_loc = os.path.join(preprocessing_path, 'XREP_test.csv') 
test_data = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)

# [CREATE DATA GENERATOR]
#==============================================================================
#==============================================================================

# initialize training device
#------------------------------------------------------------------------------
trainer = ModelTraining(device=cnf.training_device, seed=cnf.seed)

# initialize the images generator for the train and test data, and create the 
# tf.dataset according to batch shapes
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, cnf.batch_size, cnf.picture_shape, 
                                shuffle=True, augmentation=cnf.augmentation)
test_generator = DataGenerator(test_data, cnf.batch_size, cnf.picture_shape, 
                               shuffle=True, augmentation=cnf.augmentation)

# initialize the TensorDataSet class with the generator instances
# create the tf.datasets using the previously initialized generators 
datamaker = TensorDataSet()
train_dataset = datamaker.create_tf_dataset(train_generator)
test_dataset = datamaker.create_tf_dataset(test_generator)
caption_shape = datamaker.y_batch.shape[1]

# [EVALUATE XREPORT MODEL]
#==============================================================================
#==============================================================================
num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]

# Print report with info about the training parameters
#------------------------------------------------------------------------------
print(f'''
-------------------------------------------------------------------------------
XRAYREP evaluation report
-------------------------------------------------------------------------------
Number of train samples: {num_train_samples}
Number of test samples:  {num_test_samples}
-------------------------------------------------------------------------------
Batch size:              {cnf.batch_size}
Epochs:                  {cnf.epochs}
Vocabulary size:         {vocab_size + 1}
Caption length:          {caption_shape} 
-------------------------------------------------------------------------------
''')
    



