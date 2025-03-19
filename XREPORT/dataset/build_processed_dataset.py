# [SETTING ENVIRONMENT VARIABLES]
from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.process.sequences import TextSanitizer
from XREPORT.commons.utils.process.tokenizers import TokenWizard
from XREPORT.commons.utils.dataloader.serializer import DataSerializer
from XREPORT.commons.constants import CONFIG, IMG_DATA_PATH
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images     
    dataserializer = DataSerializer(CONFIG)
    dataset = dataserializer.load_dataset()    

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # sanitize text corpus by removing undesired symbols and punctuation     
    sanitizer = TextSanitizer(CONFIG)
    processed_dataset = sanitizer.sanitize_text(dataset)
    # preprocess text corpus using selected pretrained tokenizer. Text is tokenized
    # into subunits and these are eventually mapped to integer indexes        
    tokenization = TokenWizard(CONFIG) 
    logger.info(f'Tokenizing text corpus using pretrained {tokenization.tokenizer_name} tokenizer')    
    processed_dataset = tokenization.tokenize_text_corpus(processed_dataset)   
    vocabulary_size = tokenization.vocabulary_size         
    
    # 3. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer 
    logger.info(f'Dataset includes {processed_dataset.shape[0]} samples')  
    logger.info(f'Vocabulary size (unique tokens): {vocabulary_size}')
    logger.info('Saving preprocessed data to XREPORT database')     
    dataserializer.save_preprocessed_data(processed_dataset, vocabulary_size)
    

   

  

    

    

