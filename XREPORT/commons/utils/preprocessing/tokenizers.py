import pandas as pd
from transformers import DistilBertTokenizer

from XREPORT.commons.constants import CONFIG, TOKENIZER_PATH
from XREPORT.commons.logger import logger

    
# [PREPROCESSING PIPELINE]
###############################################################################
class BERTokenizer:
    
    def __init__(self):        
        
        self.max_report_size = CONFIG["dataset"]["MAX_REPORT_SIZE"]            
        self.model_identifier = 'distilbert/distilbert-base-uncased' 
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_identifier, cache_dir=TOKENIZER_PATH) 
        self.vocab_size = len(self.tokenizer.vocab)      
        logger.debug(f'Using {self.model_identifier} as tokenizer')        
    
    #--------------------------------------------------------------------------
    def BERT_tokenization(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):

        '''
        Tokenizes text data using the distilBERT Base v1.1 tokenizer. Loads the distilBERT 
        tokenizer and applies it to tokenize the provided training and validation text datasets.
        It supports padding, truncation, and returns the tokenized data in lists of token ids. 
        Additionally, it updates the source dataframes with the tokenized text.

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
    
  