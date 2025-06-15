import os

from sklearn.utils import shuffle
from transformers import AutoTokenizer

from XREPORT.commons.constants import TOKENIZERS_PATH
from XREPORT.commons.logger import logger

# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:

    def __init__(self, configuration, dataframe):

        # Set the sizes for the train and validation datasets        
        self.validation_size = configuration.get('validation_size', 1.0)
        self.seed = configuration.get('split_seed', 42)
        self.train_size = 1.0 - self.validation_size
        self.dataframe = dataframe

        # Compute the sizes of each split
        total_samples = len(dataframe)
        self.train_size = int(total_samples * self.train_size)
        self.val_size = int(total_samples * self.validation_size)
        
    #--------------------------------------------------------------------------
    def split_train_and_validation(self):
        self.dataframe = shuffle(self.dataframe, random_state=self.seed).reset_index(drop=True) 
        train_data = self.dataframe.iloc[:self.train_size]
        validation_data = self.dataframe.iloc[self.train_size:self.train_size + self.val_size]
        
        return train_data, validation_data

   
# [TOKENIZER]
###############################################################################
class TextSanitizer:
    
    def __init__(self, configuration):        
        self.max_report_size = configuration.get('max_report_size', 200)
        self.configuration = configuration    

    #--------------------------------------------------------------------------
    def sanitize_text(self, dataset):        
        dataset['text'] = dataset['text'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)        
        
        return dataset
    

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
  