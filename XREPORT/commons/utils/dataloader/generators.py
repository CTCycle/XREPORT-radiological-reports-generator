import tensorflow as tf

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DataGenerator():

    def __init__(self, configuration):         
        self.img_shape = (244, 244) # ResNet-50 input shape       
        self.augmentation = configuration["dataset"]["IMG_AUGMENT"]  
        self.batch_size = configuration["training"]["BATCH_SIZE"] 
        self.configuration = configuration 
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path, normalize=True):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)
        if self.augmentation:
            rgb_image = self.image_augmentation(rgb_image)
        if normalize:
            rgb_image = rgb_image/255.0 

        return rgb_image     
 
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
              
    
        








   


    