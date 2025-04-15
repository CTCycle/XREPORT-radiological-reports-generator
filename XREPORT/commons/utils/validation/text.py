import pandas as pd

from XREPORT.commons.utils.inference.generator import TextGenerator
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
    

# [VALIDATION OF DATA]
###############################################################################
class CalculateBLEUScore:

    def __init__(self, model, configuration : dict):
        self.model = model
        self.configuration = configuration
        self.num_samples = 100
        self.generator = TextGenerator(model, configuration)
        self.tokenizer_config = self.generator.get_tokenizer_parameters()

    #--------------------------------------------------------------------------
    def calculate_BLEU_score(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):
        sampled_train = train_data.sample(n=self.num_samples, random_state=42) 
        sampled_images = sampled_train['image'].to_list()       
        greedy_reports = self.generator.generate_radiological_reports(
            sampled_images, override_method='greedy')
        
        return greedy_reports
        