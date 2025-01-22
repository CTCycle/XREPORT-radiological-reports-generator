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
    dataset = dataserializer.get_dataset()    

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # sanitize text corpus by removing undesired symbols and punctuation     
    sanitizer = TextSanitizer(CONFIG)
    processed_dataset = sanitizer.sanitize_text(dataset)
    # preprocess text corpus using pretrained distillBERT tokenizer. Text is tokenized
    # using subwords and these are eventually mapped to integer indexes        
    tokenization = TokenWizard(CONFIG) 
    logger.info(f'Tokenizing text corpus using pretrained {tokenization.tokenizer_name} tokenizer')    
    processed_dataset = tokenization.tokenize_text_corpus(processed_dataset)   
    vocabulary_size = tokenization.vocabulary_size         
    
    # 3. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer    
    dataserializer.save_preprocessed_data(processed_dataset, vocabulary_size)
    

   

  

    

    

