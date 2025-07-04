import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.ops import concatenate

from XREPORT.commons.utils.data.process import TokenizerHandler
from XREPORT.commons.logger import logger



# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
class DataLoaderProcessor:

    def __init__(self, configuration):         
        self.img_shape = (224, 224)
        self.num_channels = 3
        # define mean and STD of pixel values for the BeiT Vision transformer  
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)    
        self.augmentation = configuration.get('use_img_augmentation', False)
        self.batch_size = configuration.get('batch_size', 32)
        self.eval_batch_size = configuration.get('eval_batch_size', 32)
        self.color_encoding = cv2.COLOR_BGR2RGB if self.num_channels==3 else cv2.COLOR_BGR2GRAY
         
        handler = TokenizerHandler(configuration)               
        self.PAD_TOKEN = handler.tokenizer.pad_token_id
        self.configuration = configuration        

    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path, as_array=False): 
        if as_array:
            image = cv2.imread(path)          
            image = cv2.cvtColor(image, self.color_encoding)
            image = np.asarray(
                cv2.resize(image, self.img_shape), dtype=np.float32)
        else:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(
                image, channels=self.num_channels, expand_animations=False)        
            image = tf.image.resize(image, self.img_shape)
        
        return image  

    # works directly with tensors
    #--------------------------------------------------------------------------
    def load_data_for_training(self, path, text):        
        rgb_image = self.load_image(path)
        rgb_image = self.image_augmentation(rgb_image) if self.augmentation else rgb_image
        rgb_image = self.image_normalization(rgb_image)        
        pad_token = tf.cast(self.PAD_TOKEN, text.dtype)
        pad_token_tf = tf.expand_dims(pad_token, 0)        
        input_text = tf.concat([text[:-1], pad_token_tf], axis=0)
        output_text = tf.concat([pad_token_tf, text[1:]], axis=0)
        
        return (rgb_image, input_text), output_text                  
 
    #--------------------------------------------------------------------------
    def load_data_for_inference(self, path, text):
        rgb_image = self.load_image(path)        
        rgb_image = self.image_normalization(rgb_image)        
        
        return rgb_image    

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
    

        

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class XRAYDataLoader:

    def __init__(self, configuration : dict, shuffle=True):        
        self.processor = DataLoaderProcessor(configuration)          
        self.batch_size = configuration.get('batch_size', 32)
        self.eval_batch_size = configuration.get('eval_batch_size', 32)
        self.shuffle_samples = configuration.get('shuffle_size', 1024)
        self.buffer_size = tf.data.AUTOTUNE   
        self.configuration = configuration
        self.shuffle = shuffle  

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def build_training_dataloader(self, data, batch_size=None, buffer_size=tf.data.AUTOTUNE):           
        images, tokens = data['path'].to_list(), data['tokens'].to_list()        
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices((images, tokens))                 
        dataset = dataset.map(
            self.processor.load_data_for_training, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=self.shuffle_samples) if self.shuffle else dataset 

        return dataset
        
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def build_inference_dataloader(self, data, batch_size=None, buffer_size=tf.data.AUTOTUNE):           
        images, tokens = data['path'].to_list(), data['tokens'].to_list()        
        batch_size = self.eval_batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices((images, tokens))                 
        dataset = dataset.map(
            self.processor.load_data_for_inference, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)        

        return dataset



   


    