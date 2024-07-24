import numpy as np
import tensorflow as tf

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DataGenerator():

    def __init__(self):        
        
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]       
        self.normalization = CONFIG["dataset"]["IMG_NORMALIZE"]
        self.augmentation = CONFIG["dataset"]["IMG_AUGMENT"]  
        self.batch_size = CONFIG["training"]["BATCH_SIZE"]  
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=1, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape[:-1])
        if self.augmentation:
            rgb_image = self.image_augmentation(rgb_image)
        if self.normalization:
            rgb_image = rgb_image/255.0 

        return rgb_image 
    
    # ...
    #--------------------------------------------------------------------------
    def process_data(self, path, text): 

        rgb_image = self.load_image(path) 
        input_text = text[:-1]
        output_text = text[1:]      

        return (rgb_image, input_text), output_text     

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):
        pp_image = tf.keras.preprocessing.image.random_shift(image, 0.2, 0.3)
        pp_image = tf.image.random_flip_left_right(pp_image)
        pp_image = tf.image.random_flip_up_down(pp_image)

        return pp_image 
              
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    ###############################################################################
    def build_tensor_dataset(self, paths, tokens, buffer_size=tf.data.AUTOTUNE):

        num_samples = len(paths) 
        paths = tf.convert_to_tensor(paths)
        tokens = tf.convert_to_tensor(tokens)

        dataset = tf.data.Dataset.from_tensor_slices((paths, tokens))
        dataset = dataset.shuffle(buffer_size=num_samples)  
        # map preprocessing function
        dataset = dataset.map(self.process_data, num_parallel_calls=buffer_size)   
        # batch and prefetch dataset
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)

        return dataset
        

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
def training_data_pipeline(train_data, validation_data):    
        
        generator = DataGenerator()                 

        train_img_paths = train_data['path'].to_list()
        train_reports = train_data['tokens'].to_list()
        val_img_paths = validation_data['path'].to_list()
        val_reports = validation_data['tokens'].to_list()

        train_dataset = generator.build_tensor_dataset(train_img_paths, train_reports)
        validation_dataset = generator.build_tensor_dataset(val_img_paths, val_reports)
        # logging debug info about batch shapes
        for (x1, x2), y in train_dataset.take(1):
            logger.debug(f'X batch shape is: {x1.shape}')  
            logger.debug(f'Y batch shape is: {y.shape}') 

        return train_dataset, validation_dataset






   


    