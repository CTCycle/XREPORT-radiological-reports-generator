import numpy as np
import pandas as pd
import tensorflow as tf

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DatasetGenerator():

    def __init__(self, configuration):        
        
        self.img_shape = configuration["model"]["IMG_SHAPE"]        
        self.augmentation = configuration["dataset"]["IMG_AUGMENT"]  
        self.batch_size = configuration["training"]["BATCH_SIZE"] 
        self.configuration = configuration 
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path, normalize=True):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=1, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape[:-1])
        if self.augmentation:
            rgb_image = self.image_augmentation(rgb_image)
        if normalize:
            rgb_image = rgb_image/255.0 

        return rgb_image 
    
    # ...
    #--------------------------------------------------------------------------
    def process_data(self, path, text): 

        rgb_image = self.load_image(path, normalize=True) 
        input_text = text[:-1]
        output_text = text[1:]      

        return (rgb_image, input_text), output_text     

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):  
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image) 

        return image
              
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def build_dataset_from_generator(self, paths, tokens, batch_size=None, buffer_size=tf.data.AUTOTUNE):

        num_samples = len(paths)  
        if batch_size is None:
            batch_size = self.configuration["training"]["BATCH_SIZE"]

        dataset = tf.data.Dataset.from_tensor_slices((paths, tokens))                 
        dataset = dataset.map(self.process_data, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples) 

        return dataset
        

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
def ML_model_dataloader(train_data : pd.DataFrame, validation_data : pd.DataFrame, 
                        configuration, batch_size=None):    
        
    generator = DatasetGenerator(configuration) 
    train_dataset = generator.build_dataset_from_generator(train_data['path'].to_list(), 
                                                           train_data['tokens'].to_list(),
                                                           batch_size)
    validation_dataset = generator.build_dataset_from_generator(validation_data['path'].to_list(), 
                                                                validation_data['tokens'].to_list(),
                                                                batch_size)       
    for (x1, x2), y in train_dataset.take(1):
        logger.debug(f'X batch shape is: {x1.shape}')  
        logger.debug(f'Y batch shape is: {y.shape}') 

    return train_dataset, validation_dataset






   


    