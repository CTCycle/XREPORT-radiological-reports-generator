import os
import pandas as pd
from transformers import AutoTokenizer

from XREPORT.commons.constants import CONFIG, TOKENIZERS_PATH
from XREPORT.commons.logger import logger


# [TOKENIZERS]
###############################################################################
class PretrainedTokenizers:

    def __init__(self): 

        self.tokenizer_strings = {'distilbert': 'distilbert/distilbert-base-uncased',
                                  'bert': 'bert-base-uncased',
                                  'roberta': 'roberta-base',
                                  'gpt2': 'gpt2',
                                  'xlm': 'xlm-mlm-enfr-1024'}
    
    #--------------------------------------------------------------------------
    def get_tokenizer(self, tokenizer_name):

        if tokenizer_name not in self.tokenizer_strings:
            tokenizer_string = tokenizer_string
            logger.warning(f'{tokenizer_string} is not among preselected models.')
        else:
            tokenizer_string = self.tokenizer_strings[tokenizer_name]                            
        
        logger.info(f'Loading {tokenizer_string} for text tokenization...')
        tokenizer_path = os.path.join(TOKENIZERS_PATH, tokenizer_name)
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_string, cache_dir=tokenizer_path)
        vocabulary_size = len(tokenizer.vocab)            

        return tokenizer, vocabulary_size 

    
# [TOKENIZER]
###############################################################################
class TokenWizard:
    
    def __init__(self, configuration):           
        tokenizer_name = configuration["dataset"]["TOKENIZER"] 
        self.max_report_size = configuration["dataset"]["MAX_REPORT_SIZE"] 
        selector = PretrainedTokenizers()
        self.tokenizer, self.vocabulary_size = selector.get_tokenizer(tokenizer_name)         
    
    #--------------------------------------------------------------------------
    def tokenize_text_corpus(self, data : pd.DataFrame):        
        # tokenize train and validation text using loaded tokenizer 
        text = data['text'].to_list()      
        tokens = self.tokenizer(text, padding=True, truncation=True,
                                max_length=self.max_report_size, return_tensors='pt')             
        
        # extract only token ids from the tokenizer output
        tokens = tokens['input_ids'].numpy().tolist()
        # add tokenizer sequences to the source dataframe, now containing the paths,
        # original text and tokenized text
        data['tokens'] = [' '.join(map(str, ids)) for ids in tokens]         
        
        return data
  