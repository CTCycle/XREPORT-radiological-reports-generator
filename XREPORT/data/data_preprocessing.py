import os
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.preprocessing.tokenizers import BERTokenizer
from XREPORT.commons.utils.dataloader.serializer import get_images_from_dataset, DataSerializer
from XREPORT.commons.utils.preprocessing.splitting import DatasetSplit
from XREPORT.commons.pathfinder import IMG_DATA_PATH, TABLES_PATH, BERT_PATH
from XREPORT.commons.configurations import SAMPLE_SIZE


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images 
    file_loc = os.path.join(TABLES_PATH, 'XREP_dataset.csv') 
    dataset = pd.read_csv(file_loc, encoding = 'utf-8', sep =';', low_memory=False)
    dataset = get_images_from_dataset(IMG_DATA_PATH, dataset, sample_size=SAMPLE_SIZE)

    # split data
    print('\nPreparing dataset of images based on splitting sizes')  
    splitter = DatasetSplit(dataset)     
    train_data, validation_data, test_data = splitter.split_data()       

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # preprocess text corpus using pretrained distillBERT tokenizer. Text is tokenized
    # using subwords and these are eventually mapped to integer indexes 
    tokenization = BERTokenizer(train_data, validation_data, path=BERT_PATH)    
    train_tokens, validation_tokens = tokenization.BERT_tokenization()
    tokenizer = tokenization.tokenizer
    vocab_size = tokenization.vocab_size

    # save preprocessed data
    dataserializer = DataSerializer()
    dataserializer.save_preprocessed_data(tokenization.train_data, 
                                          tokenization.validation_data, 
                                          TABLES_PATH)

    print(f'\nData has been succesfully preprocessed and saved in {TABLES_PATH}') 
    

   

  

    

    

