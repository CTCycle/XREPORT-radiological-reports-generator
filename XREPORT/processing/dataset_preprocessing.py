# [SET KERAS BACKEND]
import os 

# [IMPORT LIBRARIES]
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.process.sequences import TextSanitizer
from XREPORT.commons.utils.process.tokenizers import TokenWizard
from XREPORT.commons.utils.dataloader.serializer import get_images_from_dataset, DataSerializer

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
    # sanitize text corpus by removing undesired symbols and punctuation     
    sanitizer = TextSanitizer(CONFIG)
    processed_dataset = sanitizer.sanitize_text(dataset)
    # preprocess text corpus using pretrained distillBERT tokenizer. Text is tokenized
    # using subwords and these are eventually mapped to integer indexes     
    tokenization = TokenWizard(CONFIG)    
    processed_dataset = tokenization.tokenize_text_corpus(processed_dataset)   
    vocabulary_size = tokenization.vocabulary_size         
    
    # 3. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer
    dataserializer = DataSerializer(CONFIG)
    dataserializer.save_preprocessed_data(processed_dataset, vocabulary_size)
    

   

  

    

    

