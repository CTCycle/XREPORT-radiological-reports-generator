from transformers import DistilBertTokenizer

    
# [PREPROCESSING PIPELINE]
#------------------------------------------------------------------------------
class BERTokenizer:  


    def __init__(self):

        self.model_identifier = 'distilbert/distilbert-base-uncased'  

    #--------------------------------------------------------------------------
    def get_BERT_tokenizer(self, path):

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
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_identifier, cache_dir=path) 

        return tokenizer      
    
    #--------------------------------------------------------------------------
    def BERT_tokenization(self, train_text, test_text=None, path=None):

        '''
        Tokenizes text data using the BioBERT Base v1.1 tokenizer. Loads the BioBERT 
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
        
        print('\nLoading BERT tokenizer\n')        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_identifier, cache_dir=path)        
        train_tokens = self.tokenizer(train_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        if test_text is not None:
            test_tokens = self.tokenizer(test_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        else:
            test_tokens = None
        
        self.vocab_size = len(self.tokenizer.vocab)        
        
        return train_tokens, test_tokens