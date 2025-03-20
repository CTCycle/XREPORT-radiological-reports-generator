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
    def count_words_in_documents(self, data : pd.DataFrame):         
        words = [word for text in data['text'].to_list() for word in text.split()]        
           

        return words