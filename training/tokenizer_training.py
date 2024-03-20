import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

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
from utils.token_assets import *
from utils.callbacks import RealTimeHistory
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
bpe_path = os.path.join(globpt.train_path, 'BPE tokenizer')
os.mkdir(bpe_path) if not os.path.exists(bpe_path) else None


# [LOAD DATA]
#==============================================================================
#==============================================================================

# load text data from csv to generate text corpus
#------------------------------------------------------------------------------
file_loc = os.path.join(globpt.data_path, 'XREP_dataset.csv') 
dataset = pd.read_csv(file_loc, encoding='utf-8', sep=';', low_memory=False)
text_corpus = dataset['text'].to_list()

# [TRAIN TOKENIZER]
#==============================================================================
#==============================================================================

# pretrain BPE algorithm on the entire text corpus
#------------------------------------------------------------------------------
tokenizer = BPE_Tokenizer(max_vocab_size=None)  
tokenizer.train(text_corpus)

# Example of encoding
encoded_text = tokenizer.encode("This is a simple example.")
print(encoded_text)






# [BUILD XREPORT MODEL]
#==============================================================================
#==============================================================================

# Print report with info about the training parameters
#------------------------------------------------------------------------------
print(f'''
-------------------------------------------------------------------------------
XRAYREP training report
-------------------------------------------------------------------------------
Number of train samples: {cnf.num_train_samples}
Number of test samples:  {cnf.num_test_samples}
-------------------------------------------------------------------------------
Batch size:              {cnf.batch_size}
Epochs:                  {cnf.epochs}
Vocabulary size:         {vocab_size + 1}
Caption length:          {caption_shape} 
-------------------------------------------------------------------------------
''')






