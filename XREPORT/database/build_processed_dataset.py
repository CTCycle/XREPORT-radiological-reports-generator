# [SETTING ENVIRONMENT VARIABLES]
from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.data.serializer import DataSerializer
from XREPORT.commons.utils.data.process.splitting import TrainValidationSplit
from XREPORT.commons.utils.data.process.sequences import TextSanitizer
from XREPORT.commons.utils.data.process.tokenizers import TokenWizard
from XREPORT.commons.constants import CONFIG, IMG_PATH
from XREPORT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images     
    dataserializer = DataSerializer(CONFIG)
    dataset = dataserializer.load_source_dataset()    

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # sanitize text corpus by removing undesired symbols and punctuation     
    sanitizer = TextSanitizer(CONFIG)
    processed_data = sanitizer.sanitize_text(dataset)
    logger.info(f'Dataset includes {processed_data.shape[0]} samples')

    # preprocess text corpus using selected pretrained tokenizer. Text is tokenized
    # into subunits and these are eventually mapped to integer indexes        
    tokenization = TokenWizard(CONFIG) 
    logger.info(f'Tokenizing text corpus using pretrained {tokenization.tokenizer_name} tokenizer')    
    processed_data = tokenization.tokenize_text_corpus(processed_data)   
    vocabulary_size = tokenization.vocabulary_size 
    logger.info(f'Vocabulary size (unique tokens): {vocabulary_size}') 

    # 4. [SPLIT BASED NORMALIZATION AND ENCODING]
    #-------------------------------------------------------------------------- 
    # split data into train set and validation set
    logger.info('Preparing dataset of images and captions based on splitting size')  
    splitter = TrainValidationSplit(CONFIG, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation()         
    
    # 3. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------        
    dataserializer.save_train_and_validation_data(
        train_data, validation_data, vocabulary_size) 
    logger.info('Preprocessed data saved into XREPORT database')   

  

    

    

