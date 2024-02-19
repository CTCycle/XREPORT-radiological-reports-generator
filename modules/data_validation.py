import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# setting warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path if this file is launched as __main__
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

# import modules and components
#------------------------------------------------------------------------------
from modules.components.data_assets import PreProcessing, DataValidation
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD DATA]
#==============================================================================
# Load the csv with data and transform the tokenized text column to convert the
# strings into a series of integers
#==============================================================================
print('''
-------------------------------------------------------------------------------
Data Validation
-------------------------------------------------------------------------------
''')

preprocessor = PreProcessing()

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

# [DATA EVALUATION]
#==============================================================================
# ...
#==============================================================================
print('''Generating pixel intensity histograms (train vs test datasets)
''')

validator = DataValidation()

# load train and test images as numpy arrays
#------------------------------------------------------------------------------
train_images = preprocessor.load_images(train_data['images_path'], cnf.picture_shape[:-1], 
                                        as_tensor=False,  normalize=False)
test_images = preprocessor.load_images(test_data['images_path'], cnf.picture_shape[:-1], 
                                       as_tensor=False, normalize=False)

# validate pixel intensity histograms for both datasets
#------------------------------------------------------------------------------
validator.pixel_intensity_histograms(train_images, test_images, GlobVar.val_path,
                                     names=['Train', 'Test'])

# Print report with info about the data evaluation 
#------------------------------------------------------------------------------
print(f'''
-------------------------------------------------------------------------------
Data validation report
-------------------------------------------------------------------------------
Number of train samples: {train_data.shape[0]}
Number of test samples:  {test_data.shape[0]}
-------------------------------------------------------------------------------
''')
