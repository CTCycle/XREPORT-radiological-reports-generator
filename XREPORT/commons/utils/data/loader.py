import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from XREPORT.commons.utils.data.process.runtime import TrainingDataLoaderProcessor
from XREPORT.commons.logger import logger
        

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TrainingDataLoader:

    def __init__(self, configuration, shuffle=True):        
        self.processor = TrainingDataLoaderProcessor(configuration)
        self.configuration = configuration
        self.batch_size = self.configuration['training']["BATCH_SIZE"]         
        self.shuffle = shuffle   

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, data : pd.DataFrame, batch_size, buffer_size=tf.data.AUTOTUNE):             
        num_samples = data.shape[0]
        images, tokens = data['path'].to_list(), data['tokens'].to_list()        
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices((images, tokens))                 
        dataset = dataset.map(
            self.processor.load_and_process_data, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples) if self.shuffle else dataset 

        return dataset
        
    #--------------------------------------------------------------------------
    def build_training_dataloader(self, train_data : pd.DataFrame, validation_data : pd.DataFrame, 
                                  batch_size=None):        
        train_dataset = self.compose_tensor_dataset(train_data, batch_size)
        validation_dataset = self.compose_tensor_dataset(validation_data, batch_size)      
        
        return train_dataset, validation_dataset


# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class InferenceDataLoader:

    def __init__(self, configuration):
        self.processor = TrainingDataLoaderProcessor(configuration) 
        self.configuration = configuration
        self.batch_size = configuration['validation']["BATCH_SIZE"]
        self.img_shape = (224, 224, 3)
        self.num_channels = self.img_shape[-1]           
        self.color_encoding = cv2.COLOR_BGR2RGB if self.num_channels==3 else cv2.COLOR_BGR2GRAY 

    #--------------------------------------------------------------------------
    def load_image_as_array(self, path, normalization=True):       
        image = cv2.imread(path)          
        image = cv2.cvtColor(image, self.color_encoding)
        image = np.asarray(
            cv2.resize(image, self.img_shape[:-1]), dtype=np.float32)            
        if normalization:
            image = image/255.0       

        return image 

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, data : pd.DataFrame, batch_size, buffer_size=tf.data.AUTOTUNE):         
        images, tokens = data['path'].to_list(), data['tokens'].to_list()        
        batch_size = self.configuration["training"]["BATCH_SIZE"] if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices((images, tokens))                 
        dataset = dataset.map(
            self.processor.load_data, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        
        return dataset
        
    #--------------------------------------------------------------------------
    def build_inference_dataloader(self, train_data, batch_size=None):       
        dataset = self.compose_tensor_dataset(train_data, batch_size)             

        return dataset  


   


    