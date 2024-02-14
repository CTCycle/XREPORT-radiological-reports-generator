import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

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
import modules.global_variables as GlobVar
import configurations as cnf

# [CREATE TRAINING AND TEST DATASETS]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
XRAY data preprocessing
-------------------------------------------------------------------------------
The XRAY dataset must be preprocessed before feeding it to the training model.
The preprocessing procedure comprises the tokenization and padding on the text sequences
''')

# create model folder
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
model_folder = preprocessor.model_savefolder(GlobVar.models_path, 'XREP')
GlobVar.model_folder_path = model_folder
GlobVar.model_folder_name = preprocessor.folder_name

# load data from csv, add paths to images 
#------------------------------------------------------------------------------
file_loc = os.path.join(GlobVar.data_path, 'XREP_dataset.csv') 
dataset = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory = False)
dataset = preprocessor.images_pathfinder(GlobVar.images_path, dataset, 'id')

# select subset of data
#------------------------------------------------------------------------------
total_samples = cnf.num_train_samples + cnf.num_test_samples
dataset = dataset.sample(n=total_samples, random_state=cnf.seed)

# split data into train and test dataset and start preprocessor
#------------------------------------------------------------------------------
test_size = cnf.num_test_samples/total_samples
train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=cnf.seed)

# [PREPROCESS DATA]
#==============================================================================
# module for the selection of different operations
#==============================================================================

# create subfolder for preprocessing data
#------------------------------------------------------------------------------
pp_path = os.path.join(model_folder, 'preprocessing')
if not os.path.exists(pp_path):
    os.mkdir(pp_path)

# clean text corpus
#------------------------------------------------------------------------------
train_text, test_text = train_data['text'].to_list(), test_data['text'].to_list()
train_text = preprocessor.text_preparation(train_text)
test_text = preprocessor.text_preparation(test_text)

# extract text sequences and perform tokenization
#------------------------------------------------------------------------------
print('''STEP 1 ----> Text tokenization using word tokenizer
''')

tokenized_train_text, tokenized_test_text = preprocessor.text_tokenization(train_text, test_text, pp_path)

# perform padding of sequences
#------------------------------------------------------------------------------
print('''STEP 2 ----> Sequence padding to equalize sequence length
''')

pad_length = max([len(x.split()) for x in train_text])
padded_train_text = preprocessor.sequence_padding(tokenized_train_text, pad_length, output='string')
padded_test_text = preprocessor.sequence_padding(tokenized_test_text, pad_length, output='string')
train_data['tokenized_text'] = padded_train_text
test_data['tokenized_text'] = padded_test_text

# [SAVE CSV DATA]
#==============================================================================
# module for the selection of different operations
#==============================================================================
file_loc = os.path.join(pp_path, 'XREP_train.csv')  
train_data.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(pp_path, 'XREP_test.csv')  
test_data.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')


