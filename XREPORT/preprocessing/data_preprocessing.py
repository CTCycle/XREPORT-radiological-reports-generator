# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [IMPORT LIBRARIES]
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.preprocessing.tokenizers import BERTokenizer
from XREPORT.commons.utils.dataloader.serializer import get_images_from_dataset, DataSerializer
from XREPORT.commons.utils.preprocessing.splitting import DatasetSplit
from XREPORT.commons.constants import CONFIG, DATA_PATH, IMG_DATA_PATH
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images     
    dataset = get_images_from_dataset(IMG_DATA_PATH)

    # split data into train set and validation set
    logger.info('Preparing dataset of images based on splitting size')  
    splitter = DatasetSplit(dataset)     
    train_data, validation_data = splitter.split_train_and_validation()       

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # preprocess text corpus using pretrained distillBERT tokenizer. Text is tokenized
    # using subwords and these are eventually mapped to integer indexes 
    logger.info('Loading distilBERT tokenizer and apply tokenization')     
    tokenization = BERTokenizer()    
    train_data, validation_data = tokenization.BERT_tokenization(train_data, validation_data)
    
    # 3. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer
    dataserializer = DataSerializer()
    dataserializer.save_preprocessed_data(train_data, validation_data)
    

   

  

    

    

