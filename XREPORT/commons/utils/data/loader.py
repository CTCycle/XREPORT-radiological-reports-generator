import pandas as pd
import tensorflow as tf

from XREPORT.commons.utils.data.process.runtime import TrainingDataLoaderProcessor
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
        

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TrainingDataLoader:

    def __init__(self, configuration, shuffle=True):
        self.generator = TrainingDataLoaderProcessor(configuration) 
        self.configuration = configuration
        self.shuffle = shuffle   

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, data : pd.DataFrame, batch_size, buffer_size=tf.data.AUTOTUNE):             
        num_samples = data.shape[0]
        images, tokens = data['path'].to_list(), data['tokens'].to_list()        
        batch_size = self.configuration["training"]["BATCH_SIZE"] if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices((images, tokens))                 
        dataset = dataset.map(self.generator.process_data, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples) if self.shuffle else dataset 

        return dataset
        
    #--------------------------------------------------------------------------
    def build_model_dataloader(self, train_data : pd.DataFrame, validation_data : pd.DataFrame, 
                               batch_size=None):        
        train_dataset = self.compose_tensor_dataset(train_data, batch_size)
        validation_dataset = self.compose_tensor_dataset(validation_data, batch_size)      
        
        return train_dataset, validation_dataset






   


    