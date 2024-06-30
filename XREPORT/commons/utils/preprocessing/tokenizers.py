import os
import pandas as pd
from transformers import DistilBertTokenizer

    
# [PREPROCESSING PIPELINE]
#------------------------------------------------------------------------------
class BERTokenizer:  


    def __init__(self, train_data : pd.DataFrame, validation_data : pd.DataFrame, path=''):

        self.path = path
        self.train_data = train_data
        self.validation_data = validation_data
        
        self.train_text = train_data['text'].to_list()
        self.validation_text = validation_data['text'].to_list()
            
        self.model_identifier = 'distilbert/distilbert-base-uncased'  

    #--------------------------------------------------------------------------
    def get_BERT_tokenizer(self):

        '''
        Loads and initializes the BioBERT Base v1.1 tokenizer. It optionally
        accepts a path to cache the tokenizer.

        Keyword Arguments:
            path (str, optional): Directory path to cache the tokenizer. 
                                  If not specified, the default cache location is used.

        Returns:
            tokenizer (AutoTokenizer): The loaded BioBERT tokenizer.

        '''
        
        print('\nLoading BERT tokenizer\n')        
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_identifier, cache_dir=self.path) 

        return tokenizer      
    
    #--------------------------------------------------------------------------
    def BERT_tokenization(self):

        '''
        Tokenizes text data using the distilBERT Base v1.1 tokenizer. Loads the distilBERT 
        tokenizer and applies it to tokenize the provided training (and optionally testing)
        text datasets. It supports padding, truncation, and returns the tokenized data 
        in TensorFlow tensors. 

        Keyword Arguments:
            train_text (list of str): The text data for training to tokenize.
            test_text (list of str, optional): The text data for testing to tokenize. 
                                               Default is None, indicating no test text.
            path (str, optional): Path to cache the BioBERT tokenizer. 
                                  Default is None, using the default cache directory.

        Returns:
            tuple: A tuple containing two elements:
                - train_tokens (tf.Tensor): Tokenized version of `train_text`.
                - test_tokens (tf.Tensor or None): Tokenized version of `test_text` if provided, otherwise None.

        '''         
        print('\nLoading distilBERT tokenizer and apply tokenization\n')        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_identifier, cache_dir=self.path) 
        
               
        train_tokens = self.tokenizer(self.train_text, padding=True, 
                                      truncation=True, 
                                      max_length=200, 
                                      return_tensors='tf')
        validation_tokens = self.tokenizer(self.validation_text, padding=True, 
                                           truncation=True, 
                                           max_length=200, 
                                           return_tensors='tf')        
        
        self.vocab_size = len(self.tokenizer.vocab)
        
        # extract only token ids from the tokenizer output
        train_tokens = train_tokens['input_ids'].numpy().tolist() 
        val_tokens = validation_tokens['input_ids'].numpy().tolist()

        # add tokenizer sequences to the source dataframe, now containing the paths,
        # original text and tokenized text
        self.train_data['tokens'] = [' '.join(map(str, ids)) for ids in train_tokens]  
        self.validation_data['tokens'] = [' '.join(map(str, ids)) for ids in val_tokens]        
        
        return train_tokens, val_tokens
    
  