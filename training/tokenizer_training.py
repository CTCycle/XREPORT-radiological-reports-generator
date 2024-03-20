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
from utils.token_assets import BPETokenizer
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
text_corpus_fragments = dataset['text'].to_list()
text_corpus = ' '.join(text_corpus_fragments)

# [TRAIN TOKENIZER]
#==============================================================================
#==============================================================================

# pretrain BPE algorithm on the entire text corpus
#------------------------------------------------------------------------------
tokenizer = BPETokenizer()  
tokenizer.learn_bpe(text_corpus, num_merges=5000)
tokenizer.finalize_vocab()

# Example of encoding
encoded_text = tokenizer.apply_bpe('This is a simple example.')
vectorized_text = tokenizer.vectorize('This is a simple example.')
print(encoded_text)
print(vectorized_text)

# save tokenizer
tokenizer.save(bpe_path)









