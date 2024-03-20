import os
import json
from collections import Counter, defaultdict
from transformers import AutoTokenizer, BertTokenizer
from tqdm import tqdm



# [TOOLKIT TO USE THE PRETRAINED MODEL]
#==============================================================================
# Custom training operations
#==============================================================================
class BPETokenizer:

    

    def __init__(self, vocab=None, bpe_merges=None):
        self.vocab = vocab if vocab is not None else {}
        self.bpe_merges = bpe_merges if bpe_merges is not None else []

    #--------------------------------------------------------------------------
    def build_vocab(self, text):       
        
        words = text.split()
        vocab = Counter(words)
        return {' '.join(word) + ' </w>': count for word, count in vocab.items()}

    #--------------------------------------------------------------------------
    def get_stats(self, vocab):
        
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
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
    def learn_bpe(self, text, num_merges=100):
       
        self.vocab = self.build_vocab(text)
        for i in tqdm(range(num_merges)):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            self.bpe_merges.append(best)

    #--------------------------------------------------------------------------
    def apply_bpe(self, text):
        
        words = text.split()
        tokens = []
        for word in words:
            word = ' '.join(word) + ' </w>'
            for bpe_merge in self.bpe_merges:
                while True:
                    bigram = ' '.join(bpe_merge)
                    if bigram in word:
                        word = word.replace(bigram, ''.join(bpe_merge))
                    else:
                        break
            tokens.extend(word.split())
        
        self.finalize_vocab()

        return tokens

    #--------------------------------------------------------------------------
    def finalize_vocab(self):
        
        subwords = set()
        for word in self.vocab:
            subwords.update(word.split())        
        self.subword_to_id = {subword: i + 1 for i, subword in enumerate(sorted(subwords))}

    #--------------------------------------------------------------------------
    def vectorize(self, text):
        
        tokens = self.apply_bpe(text)
        return [self.subword_to_id[token] for token in tokens if token in self.subword_to_id]

    #--------------------------------------------------------------------------
    def save(self, path):
       
        filepath = os.path.join(path, 'BPE_tokenizer.json')
        with open(filepath, 'w') as f:
            json.dump({'vocab': self.vocab,
                       'bpe_merges': self.bpe_merges,
                       'subword_to_id': self.subword_to_id}, f, indent=4)

    #--------------------------------------------------------------------------
    @classmethod
    def load(cls, path):
       
        filepath = os.path.join(path, 'BPE_tokenizer.json')
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(vocab=data['vocab'], bpe_merges=data['bpe_merges'], subword_to_id=data['subword_to_id'])

    

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