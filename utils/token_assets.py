import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras import layers
from transformers import TFAutoModelForMaskedLM
from transformers import AutoTokenizer, BertTokenizer, AutoModel



# [TOOLKIT TO USE THE PRETRAINED MODEL]
#==============================================================================
# Custom training operations
#==============================================================================
class BPE_Tokenizer:
    def __init__(self, max_vocab_size=1000):        
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.bpe_codes = {}
    
    #--------------------------------------------------------------------------
    def get_stats(self, vocabulary):
        pairs = {}
        for word, freq in vocabulary.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                current_freq = pairs.get(pair, 0)
                pairs[pair] = current_freq + freq
        return pairs

    #--------------------------------------------------------------------------
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = ' '.join(pair)
        replacer = ''.join(pair)
        for word in v_in:
            w_out = word.replace(bigram, replacer)
            v_out[w_out] = v_in[word]
        return v_out

    #--------------------------------------------------------------------------
    def build_vocab(self, text):
        vocab = {}
        for word in text.split(' '):
            word = ' '.join(list(word)) + ' </w>'  
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        return vocab

    #--------------------------------------------------------------------------
    def train(self, text):
        self.vocab = self.build_vocab(text)
        for i in range(self.vocab_size):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            self.bpe_codes[best] = i

    #--------------------------------------------------------------------------
    def encode(self, text):
        text = ' '.join(list(text)) + ' </w>'
        for bpe_code in sorted(self.bpe_codes, key=lambda x: self.bpe_codes[x]):
            if bpe_code in text:
                text = text.replace(' '.join(bpe_code), ''.join(bpe_code))
        return text.split()
    

# [TOOLKIT TO USE THE PRETRAINED MODEL]
#==============================================================================
# Custom training operations
#==============================================================================
class BERT_hub:

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
        model_identifier = 'google-bert/bert-base-uncased'
        print('\nLoading BERT tokenizer\n')        
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, cache_dir=path) 

        return tokenizer      
    
    #--------------------------------------------------------------------------
    def BPE_tokenization(self, train_text, test_text=None, path=None):

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
        model_identifier = 'google-bert/bert-base-uncased'
        print('\nLoading BERT subword tokenizer\n')        
        self.tokenizer = BertTokenizer.from_pretrained(model_identifier, cache_dir=path)        
        train_tokens = self.tokenizer(train_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        if test_text is not None:
            test_tokens = self.tokenizer(test_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        else:
            test_tokens = None        
        self.vocab_size = len(self.tokenizer.vocab)        
        
        return train_tokens, test_tokens