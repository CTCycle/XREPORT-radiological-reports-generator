import os
import sys
import numpy as np
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
from modules.components.training_assets import ModelTraining, RealTimeHistory, DataGenerator, XREPCaptioningModel
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD DATA AND ADD IMAGES PATHS TO DATASET]
#==============================================================================
# Load the csv with data and transform the tokenized text column to convert the
# strings into a series of integers
#==============================================================================
print('''
-------------------------------------------------------------------------------
XRAYREP training
-------------------------------------------------------------------------------
XRAYREP model will be trained on the preprocessed data...
''')

file_loc = os.path.join(GlobVar.data_path, 'XREP_train.csv') 
df_train = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)
file_loc = os.path.join(GlobVar.data_path, 'XREP_test.csv') 
df_test = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)


# assign paths to images in the dataset
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
df_train = preprocessor.images_pathfinder(GlobVar.images_path, df_train, 'id')
df_test = preprocessor.images_pathfinder(GlobVar.images_path, df_test, 'id')

# initialize training ops and create model savefolder
#------------------------------------------------------------------------------
model_savepath = preprocessor.model_savefolder(GlobVar.model_path, 'XREP')
trainworker = ModelTraining(device = cnf.training_device)

# [ESTABLISH DATA GENERATOR]
#==============================================================================
# module for the selection of different operations
#==============================================================================

# load tokenizer to get padding length and vocabulary size
#------------------------------------------------------------------------------
tokenizer_path = os.path.join(GlobVar.data_path, 'Tokenizers')
tokenizer = preprocessor.load_tokenizer(tokenizer_path, 'word_tokenizer')
vocab_size = len(tokenizer.word_index)

# transform string sequence into list of elements
#------------------------------------------------------------------------------
train_datagen = DataGenerator(df_train, cnf.batch_size, cnf.picture_size, cnf.num_channels, shuffle=True)
test_datagen = DataGenerator(df_test, cnf.batch_size, cnf.picture_size, cnf.num_channels, shuffle=True)

# define the output signature of the generator using tf.TensorSpec
#------------------------------------------------------------------------------
x_batch, y_batch = train_datagen.__getitem__(0)
img_shape = x_batch[0].shape
tokenseq_shape = x_batch[1].shape
caption_shape = y_batch.shape
output_signature = ((tf.TensorSpec(shape=img_shape, dtype=tf.float32),
                     tf.TensorSpec(shape=tokenseq_shape, dtype=tf.float32)),
                     tf.TensorSpec(shape=caption_shape, dtype=tf.float32))

# generate tf.dataset from generator and set prefetch
#------------------------------------------------------------------------------
train_dataset = tf.data.Dataset.from_generator(lambda : train_datagen, output_signature=output_signature)
test_dataset = tf.data.Dataset.from_generator(lambda : test_datagen, output_signature=output_signature)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# [PRINT STATISTICAL REPORT]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
Number of train samples: {df_train.shape[0]}
Number of test samples:  {df_test.shape[0]}
Batch size:              {cnf.batch_size}
Vocabulary size:         {vocab_size}
Caption length:          {caption_shape[1]} 
-------------------------------------------------------------------------------
''')

# [BUILD XREPORT MODEL]
#==============================================================================
# ....
#==============================================================================

# initialize, compile and print the summary of the captioning model
#------------------------------------------------------------------------------
caption_model = XREPCaptioningModel(cnf.image_shape, caption_shape[1], vocab_size, cnf.embedding_dims,
                                    cnf.kernel_size, cnf.num_heads, cnf.learning_rate, cnf.XLA_acceleration, cnf.seed)
caption_model.compile()
caption_model.summary()

# generate graphviz plot fo the model layout
#------------------------------------------------------------------------------
if cnf.generate_model_graph == True:
    caption_model.plot_model(model_savepath)
    
# [TRAINING XREPORT MODEL]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir tensorboard/
#==============================================================================

# initialize real time plot callback
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(model_savepath, validation=True)

# initialize tensorboard
#------------------------------------------------------------------------------
if cnf.use_tensorboard == True:
    log_path = os.path.join(model_savepath, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    callbacks = [RTH_callback, tensorboard_callback]    
else:    
    callbacks = [RTH_callback]

# training loop and model saving at end
#------------------------------------------------------------------------------
print(f'''Start model training for {cnf.epochs} epochs and batch size of {cnf.batch_size}
       ''')

training = caption_model.fit(train_dataset, validation_data=test_dataset, epochs=cnf.epochs, 
                             callbacks=callbacks, workers=6, use_multiprocessing=True)                          

trainworker.save_model(caption_model, model_savepath)




