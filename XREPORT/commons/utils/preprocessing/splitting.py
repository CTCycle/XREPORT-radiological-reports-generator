import pandas as pd

from XREPORT.commons.constants import CONFIG

# [DATA SPLITTING]
#------------------------------------------------------------------------------
class DatasetSplit:

    def __init__(self, dataframe: pd.DataFrame):

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError('dataframe must be a pandas DataFrame')

        # Set the sizes for the train, validation, and test datasets
        self.test_size = CONFIG["dataset"]["TEST_SIZE"]
        self.validation_size = CONFIG["dataset"]["VALIDATION_SIZE"]
        self.train_size = 1.0 - self.test_size - self.validation_size
        self.dataframe = dataframe.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

        # Compute the sizes of each split
        total_samples = len(dataframe)
        self.train_size = int(total_samples * self.train_size)
        self.val_size = int(total_samples * self.validation_size)
        self.test_size = total_samples - self.train_size - self.val_size

    def split_data(self):

        # Split the DataFrame based on the specified sizes
        train_data = self.dataframe.iloc[:self.train_size]
        validation_data = self.dataframe.iloc[self.train_size:self.train_size + self.val_size]
        test_data = self.dataframe.iloc[self.train_size + self.val_size:]
        
        return train_data, validation_data, test_data

   
