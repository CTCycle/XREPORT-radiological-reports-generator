import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from XREPORT.commons.logger import logger



# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
class TrainingDataLoaderProcessor():

    def __init__(self, configuration):         
        self.img_shape = (224, 224)
        # define mean and STD of pixel values for the BeiT Vision transformer  
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)    
        self.augmentation = configuration["dataset"]["IMG_AUGMENTATION"]  
        self.batch_size = configuration["training"]["BATCH_SIZE"] 
        self.configuration = configuration        

    #--------------------------------------------------------------------------
    def load_image(self, path):        
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(
            image, channels=3, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)         

        return rgb_image 

    #--------------------------------------------------------------------------
    def load_data(self, path, text):        
        rgb_image = self.load_image(path)        
        rgb_image = self.image_normalization(rgb_image)        
        input_text, output_text = text[:-1], text[1:]                   

        return (rgb_image, input_text), output_text                
 
    #--------------------------------------------------------------------------
    def load_and_process_data(self, path, text):
        rgb_image = self.load_image(path)
        rgb_image = self.image_augmentation(rgb_image) if self.augmentation else rgb_image
        rgb_image = self.image_normalization(rgb_image)        
        input_text, output_text = text[:-1], text[1:]          

        return (rgb_image, input_text), output_text    

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_normalization(self, image):
        normalize_image = image/255.0        
        normalize_image = (normalize_image - self.image_mean) / self.image_std
                
        return normalize_image 

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):  
        # perform random image augmentations such as flip, brightness, contrast          
        augmentations = {"flip_left_right": (
            lambda img: tf.image.random_flip_left_right(img), 0.5),
                         "flip_up_down": (
            lambda img: tf.image.random_flip_up_down(img), 0.5),                        
                         "brightness": (
            lambda img: tf.image.random_brightness(img, max_delta=0.2), 0.25),
                         "contrast": (
            lambda img: tf.image.random_contrast(img, lower=0.7, upper=1.3), 0.35)}    
        
        for _, (func, prob) in augmentations.items():
            if np.random.rand() <= prob:
                image = func(image)
        
        return image
    
    
# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
class InferenceDataLoaderProcessor():

    def __init__(self, configuration):         
        self.img_shape = (224, 224)
        # define mean and STD of pixel values for the BeiT Vision transformer  
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32) 
        self.batch_size = configuration["validation"]["BATCH_SIZE"] 
        self.configuration = configuration        

    #--------------------------------------------------------------------------
    def load_image(self, path):        
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(
            image, channels=3, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)         

        return rgb_image        
 
    #--------------------------------------------------------------------------
    def load_and_process_data(self, path, text):
        rgb_image = self.load_image(path)        
        rgb_image = self.image_normalization(rgb_image)        
        input_text, output_text = text[:-1], text[1:]          

        return (rgb_image, input_text), output_text    

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_normalization(self, image):
        normalize_image = image/255.0        
        normalize_image = (normalize_image - self.image_mean) / self.image_std
                
        return normalize_image 

        

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TrainingDataLoader:

    def __init__(self, configuration, shuffle=True):        
        self.processor = TrainingDataLoaderProcessor(configuration)        
        self.batch_size = configuration['training']["BATCH_SIZE"] 
        self.shuffle_samples = 1024
        self.configuration = configuration        
        self.shuffle = shuffle   

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, data, batch_size, buffer_size=tf.data.AUTOTUNE):           
        images, tokens = data['path'].to_list(), data['tokens'].to_list()        
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices((images, tokens))                 
        dataset = dataset.map(
            self.processor.load_and_process_data, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=self.shuffle_samples) if self.shuffle else dataset 

        return dataset
        
    #--------------------------------------------------------------------------
    def build_training_dataloader(self, train_data, validation_data, 
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
    def compose_tensor_dataset(self, data, batch_size, buffer_size=tf.data.AUTOTUNE):         
        images, tokens = data['path'].to_list(), data['tokens'].to_list()        
        batch_size = self.configuration.get('batch_size', 32) if batch_size is None else batch_size
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


   


    