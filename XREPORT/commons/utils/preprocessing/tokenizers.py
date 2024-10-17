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
        vocab_size = len(tokenizer.vocab)            

        return tokenizer, vocab_size 

    
# [TOKENIZER]
###############################################################################
class TokenWizard:
    
    def __init__(self, configuration):        
        
        tokenizer_name = configuration["dataset"]["TOKENIZER"] 
        self.max_report_size = configuration["dataset"]["MAX_REPORT_SIZE"] 
        selector = PretrainedTokenizers()
        self.tokenizer, self.vocab_size = selector.get_tokenizer(tokenizer_name)         
    
    #--------------------------------------------------------------------------
    def tokenize_text_corpus(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):

        '''      
        Tokenizes text data using the specified tokenizer and updates the input DataFrames.

        Keyword Arguments:
            train_data (pd.DataFrame): DataFrame containing training data with a 'text' column.
            validation_data (pd.DataFrame): DataFrame containing validation data with a 'text' column.

        Returns:
            tuple: A tuple containing two elements:
                - train_data (pd.DataFrame): DataFrame with an additional 'tokens' column containing 
                  tokenized version of the 'text' column as lists of token ids.
                - validation_data (pd.DataFrame): DataFrame with an additional 'tokens' column containing 
                  tokenized version of the 'text' column as lists of token ids.        

        '''        
        self.train_text = train_data['text'].to_list()
        self.validation_text = validation_data['text'].to_list()
        
        # tokenize train and validation text using loaded tokenizer 
        train_tokens = self.tokenizer(self.train_text, padding=True, truncation=True,
                                      max_length=self.max_report_size, return_tensors='pt')
        validation_tokens = self.tokenizer(self.validation_text, padding=True, truncation=True, 
                                           max_length=self.max_report_size, return_tensors='pt')       
        
        # extract only token ids from the tokenizer output
        train_tokens = train_tokens['input_ids'].numpy().tolist() 
        val_tokens = validation_tokens['input_ids'].numpy().tolist()

        # add tokenizer sequences to the source dataframe, now containing the paths,
        # original text and tokenized text
        train_data['tokens'] = [' '.join(map(str, ids)) for ids in train_tokens]  
        validation_data['tokens'] = [' '.join(map(str, ids)) for ids in val_tokens]        
        
        return train_data, validation_data
    
  