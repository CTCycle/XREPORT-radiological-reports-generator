import os
from transformers import AutoTokenizer

from XREPORT.commons.constants import TOKENIZERS_PATH
from XREPORT.commons.logger import logger

    
# [TOKENIZER]
###############################################################################
class TokenWizard:
    
    def __init__(self, configuration):           
        self.string_id = configuration.get('tokenizer', 'distilbert') 
        self.max_report_size = configuration.get('max_report_size', 200)        
        self.tokenizer, self.vocabulary_size = self.get_tokenizer(self.string_id)   

    #--------------------------------------------------------------------------
    def get_tokenizer(self, tokenizer_name):
        tokenizer_path = os.path.join(TOKENIZERS_PATH, tokenizer_name)
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=tokenizer_path)
        vocabulary_size = len(tokenizer.vocab)            

        return tokenizer, vocabulary_size       
    
    #--------------------------------------------------------------------------
    def tokenize_text_corpus(self, data):        
        # tokenize train and validation text using loaded tokenizer 
        text = data['text'].to_list()      
        tokens = self.tokenizer(
            text, padding=True, truncation=True, 
            max_length=self.max_report_size, return_tensors='pt')             
        
        # extract only token ids from the tokenizer output
        tokens = tokens['input_ids'].numpy().tolist()
        # add tokenizer sequences to the source dataframe, now containing the paths,
        # original text and tokenized text
        data['tokens'] = [' '.join(map(str, ids)) for ids in tokens]         
        
        return data
  