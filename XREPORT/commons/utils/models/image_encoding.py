from tensorflow import keras
from keras import layers

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


      
# [IMAGE ENCODER MODEL]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='Encoders', name='ImageEncoder')
class ImageEncoder(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        self.embedding_dims = CONFIG["model"]["EMBEDDING_DIMS"] 
        self.conv1 = layers.Conv2D(128, (2,2), strides=1, padding='same', 
                                   activation='relu', kernel_initializer='he_uniform')        
        self.conv2 = layers.Conv2D(256, (2,2), strides=1, padding='same', 
                                   activation='relu', kernel_initializer='he_uniform')           
        self.conv3 = layers.Conv2D(256, (2,2), strides=1, padding='same', 
                                   activation='relu', kernel_initializer='he_uniform')  
        self.conv4 = layers.Conv2D(512, (2,2), strides=1, padding='same', 
                                   activation='relu', kernel_initializer='he_uniform')        
        self.conv5 = layers.Conv2D(512, (2,2), strides=1, padding='same', 
                                   activation='relu', kernel_initializer='he_uniform') 
        self.conv6 = layers.Conv2D(512, (2,2), strides=1, padding='same', 
                                   activation='relu', kernel_initializer='he_uniform')        
        self.maxpool1 = layers.MaxPooling2D((2,2), strides=2, padding='same')
        self.maxpool2 = layers.MaxPooling2D((2,2), strides=2, padding='same')
        self.maxpool3 = layers.MaxPooling2D((2,2), strides=2, padding='same')
        self.maxpool4 = layers.MaxPooling2D((2,2), strides=2, padding='same')          
        self.dense1 = layers.Dense(self.embedding_dims, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(self.embedding_dims, activation='relu', kernel_initializer='he_uniform')        
        self.reshape = layers.Reshape((-1, self.embedding_dims))        

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, x, training=None):              
        layer = self.conv1(x)                  
        layer = self.maxpool1(layer) 
        layer = self.conv2(layer)                     
        layer = self.maxpool2(layer)        
        layer = self.conv3(layer)  
        layer = self.conv4(layer)                        
        layer = self.maxpool3(layer)                
        layer = self.conv5(layer) 
        layer = self.conv6(layer)               
        layer = self.maxpool4(layer)         
        layer = self.dense1(layer)        
        layer = self.dense2(layer)            
        output = self.reshape(layer)              
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(ImageEncoder, self).get_config()       
        config.update({})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

