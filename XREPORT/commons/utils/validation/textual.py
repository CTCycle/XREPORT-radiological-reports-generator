import pandas as pd

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [VALIDATION OF DATA]
###############################################################################
class TextAnalysis:


    def __init__(self):
        self.DPI = 400
        self.file_type = 'jpg'

    #--------------------------------------------------------------------------
    def words_counter(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):        
        
        train_words = [word for text in train_data['text'].to_list() for word in text.split()]
        validation_words = [word for text in validation_data['text'].to_list() for word in text.split()]
        total_words = train_words + validation_words
        print(f'Number of words in the entire dataset:        {len(total_words)}')
        print(f'Number of unique words in the entire dataset: {len(set(total_words))}')
        print(f'Number of words in the training dataset:        {len(train_words)}')
        print(f'Number of unique words in the training dataset: {len(set(train_words))}')
        print(f'Number of words in the validation dataset:        {len(validation_words)}')
        print(f'Number of unique words in the validation dataset: {len(set(validation_words))}')

        return (total_words, train_words, validation_words)