# [SET KERAS BACKEND]
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [IMPORT LIBRARIES]
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.process.tokenizers import TokenWizard
from XREPORT.commons.utils.dataloader.serializer import get_images_from_dataset, DataSerializer
from XREPORT.commons.utils.process.splitting import DatasetSplit
from XREPORT.commons.constants import CONFIG, IMG_DATA_PATH
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images     
    dataset = get_images_from_dataset(IMG_DATA_PATH)

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # preprocess text corpus using pretrained distillBERT tokenizer. Text is tokenized
    # using subwords and these are eventually mapped to integer indexes          
    tokenization = TokenWizard(CONFIG)    
    dataset = tokenization.tokenize_text_corpus(dataset)   
    vocabulary_size = tokenization.vocabulary_size

    # 3. [SPLIT DATA]
    #--------------------------------------------------------------------------
    # split data into train set and validation set
    logger.info('Preparing dataset of images based on splitting size')  
    splitter = DatasetSplit(CONFIG, dataset)     
    train_data, validation_data = splitter.split_train_and_validation()       
    
    # 4. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer
    dataserializer = DataSerializer(CONFIG)
    dataserializer.save_preprocessed_data(train_data, validation_data, vocabulary_size)
    

   

  

    

    

