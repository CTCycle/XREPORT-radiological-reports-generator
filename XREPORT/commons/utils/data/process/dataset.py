from sklearn.utils import shuffle

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
    