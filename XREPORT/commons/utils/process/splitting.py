import pandas as pd

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger

# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:

    def __init__(self, configuration, dataframe: pd.DataFrame):

        # Set the sizes for the train and validation datasets        
        self.validation_size = configuration["dataset"]["VALIDATION_SIZE"]
        self.train_size = 1.0 - self.validation_size
        self.dataframe = dataframe.reset_index(drop=True)  

        # Compute the sizes of each split
        total_samples = len(dataframe)
        self.train_size = int(total_samples * self.train_size)
        self.val_size = int(total_samples * self.validation_size)
        
    #--------------------------------------------------------------------------
    def split_train_and_validation(self):

        # Split the DataFrame based on the specified sizes
        train_data = self.dataframe.iloc[:self.train_size]
        validation_data = self.dataframe.iloc[self.train_size:self.train_size + self.val_size]
        
        return train_data, validation_data

   
