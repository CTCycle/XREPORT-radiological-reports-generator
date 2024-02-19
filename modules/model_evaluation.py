import os
import sys
import pandas as pd
import tensorflow as tf

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path to sys
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

# import modules and classes
#------------------------------------------------------------------------------    
from modules.components.data_assets import PreProcessing
from modules.components.model_assets import ModelTraining, DataGenerator, Inference
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD MODEL AND DATA]
#==============================================================================
# ....
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
XREPORT model evaluation
-------------------------------------------------------------------------------
...
''') 

# Load pretrained model and its parameters
#------------------------------------------------------------------------------
inference = Inference() 
model, parameters = inference.load_pretrained_model(GlobVar.models_path)
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
df_train = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)
file_loc = os.path.join(preprocessing_path, 'XREP_test.csv') 
df_test = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)

# [CREATE DATA GENERATOR]
#==============================================================================
# initialize a custom generator to load data on the fly
#==============================================================================

# initialize training device
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
trainer = ModelTraining(device=cnf.training_device, seed=cnf.seed)

# initialize generators for X and Y subsets
#------------------------------------------------------------------------------
num_train_samples = df_train.shape[0]
num_test_samples = df_test.shape[0]
train_datagen = DataGenerator(df_train, 200, parameters['picture_shape'], 
                              shuffle=True, augmentation=False)
test_datagen = DataGenerator(df_test, 200, parameters['picture_shape'], 
                             shuffle=True, augmentation=False)

# define the output signature of the generator using tf.TensorSpec, in order to
# successfully build a tf.dataset object from the custom generator
#------------------------------------------------------------------------------
x_batch, y_batch = train_datagen.__getitem__(0)
img_shape = x_batch[0].shape
tokenseq_shape = x_batch[1].shape
caption_shape = y_batch.shape
output_signature = ((tf.TensorSpec(shape=img_shape, dtype=tf.float32),
                     tf.TensorSpec(shape=tokenseq_shape, dtype=tf.float32)),
                     tf.TensorSpec(shape=caption_shape, dtype=tf.float32))

# generate tf.dataset from the initialized generator using the output signaturs.
# set prefetch (auto-tune) on the freshly created tf.dataset
#------------------------------------------------------------------------------
df_train = tf.data.Dataset.from_generator(lambda : train_datagen, output_signature=output_signature)
df_test = tf.data.Dataset.from_generator(lambda : test_datagen, output_signature=output_signature)
df_train = df_train.prefetch(buffer_size=tf.data.AUTOTUNE)
df_test = df_test.prefetch(buffer_size=tf.data.AUTOTUNE)

# [EVALUATE XREPORT MODEL]
#==============================================================================
# ...
#==============================================================================

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
Caption length:          {caption_shape[1]} 
-------------------------------------------------------------------------------
''')
    



