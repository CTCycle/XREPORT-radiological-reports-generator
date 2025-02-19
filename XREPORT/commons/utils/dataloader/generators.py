import keras
import numpy as np
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
        self.img_shape = (224, 224) 
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)    
        self.augmentation = configuration["dataset"]["IMG_AUGMENT"]  
        self.batch_size = configuration["training"]["BATCH_SIZE"] 
        self.configuration = configuration        

    #--------------------------------------------------------------------------
    def load_image(self, path):        
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)         

        return rgb_image               
 
    #--------------------------------------------------------------------------
    def process_data(self, path, text):
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)
        rgb_image = self.image_augmentation(rgb_image) if self.augmentation else rgb_image

        input_text = text[:-1]
        output_text = text[1:]      

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
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image) 

        return image
              
    
        








   


    