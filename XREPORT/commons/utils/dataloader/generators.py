import tensorflow as tf
from PIL import Image
import numpy as np
import io

from XREPORT.commons.utils.learning.encoder import ImageEncoder
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

        encoder = ImageEncoder()
        self.processor, self.img_encoder = encoder.build_image_encoder()        

    #--------------------------------------------------------------------------
    def _py_processor(self, image_bytes):
        """
        This helper function runs in regular Python (via tf.py_function).
        It:
          1. Converts raw bytes into a PIL Image,
          2. Resizes the image to the expected shape,
          3. Runs the Hugging Face processor (with return_tensors="pt"),
          4. Returns the processed image as a NumPy array.
          
        The returned tensor is in channels-first order (C, H, W).
        """
        # image_bytes is a numpy scalar (bytes); ensure it’s a byte string
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        
        # Load image using PIL and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.img_shape)
        
        # Process image with Hugging Face processor (PyTorch backend)
        # This returns a dictionary; the key "pixel_values" holds a torch.Tensor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # shape: [1, C, H, W]
        pixel_values = pixel_values.squeeze(0)   # shape: [C, H, W]
        
        # Convert to NumPy (dtype is typically float32)
        np_image = pixel_values.numpy()
        return np_image

    #--------------------------------------------------------------------------
    def load_image(self, path):
        # Read the image file (as raw bytes) using TF ops
        image_bytes = tf.io.read_file(path)
        # Wrap our Python function with tf.py_function
        rgb_image = tf.py_function(
            func=self._py_processor,
            inp=[image_bytes],
            Tout=tf.float32)
        
        rgb_image = tf.transpose(rgb_image, perm=[1, 2, 0])        
        
        # If augmentation is enabled, we need to work in channels-last order for TF ops:
        if self.augmentation:
            # Convert to channels-last: (H, W, C)
            rgb_image = tf.transpose(rgb_image, perm=[1, 2, 0])
            rgb_image = self.image_augmentation(rgb_image)
           
        return rgb_image          
 
    #--------------------------------------------------------------------------
    def process_data(self, path, text):
        rgb_image = self.load_image(path) 
        input_text = text[:-1]
        output_text = text[1:]      

        return (rgb_image, input_text), output_text     

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):  
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image) 

        return image
              
    
        








   


    