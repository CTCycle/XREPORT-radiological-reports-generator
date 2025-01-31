import keras
from keras import activations, layers 

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [POOLING CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='StackedBatchNormConv')
class StackedBatchNormConv(layers.Layer):
    def __init__(self, units, num_layers=3, **kwargs):
        super(StackedBatchNormConv, self).__init__(**kwargs)
        self.units = units        
        self.num_layers = num_layers            
        self.convolutions = [layers.Conv2D(units, kernel_size=(2,2), padding='same') 
                             for _ in range(num_layers)]
        self.batch_norm_layers = [layers.BatchNormalization() for _ in range(num_layers)]             
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None):
        layer = inputs
        for conv, bn in zip(self.convolutions, self.batch_norm_layers):
            layer = conv(layer) 
            layer = bn(layer, training=training)
            layer = activations.relu(layer)           
        
        return layer
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(StackedBatchNormConv, self).get_config()
        config.update({'units': self.units,                  
                       'num_layers': self.num_layers})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
          

# [POOLING CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='FeedFowardBatchNorm')
class FeedFowardBatchNorm(layers.Layer):
    def __init__(self, units, num_layers=3, **kwargs):
        super(FeedFowardBatchNorm, self).__init__(**kwargs)
        self.units = units        
        self.num_layers = num_layers       
        self.dense_layers = [layers.Dense(units, activation='relu', kernel_initializer='he_uniform')
                             for _ in range(num_layers)]
        self.batch_norm = layers.BatchNormalization()                  
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None):
        layer = inputs
        for dense in self.dense_layers:
            layer = dense(layer)        
        output = self.batch_norm(layer, training=training)          
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeedFowardBatchNorm, self).get_config()
        config.update({'units': self.units,                  
                       'num_layers': self.num_layers})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)     

      
# [IMAGE ENCODER MODEL]
###############################################################################
@keras.utils.register_keras_serializable(package='Encoders', name='ImageEncoder')
class ImageEncoder(keras.Model):
    def __init__(self, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        self.embedding_dims = CONFIG["model"]["EMBEDDING_DIMS"] 
        self.convblock1 = StackedBatchNormConv(64, 2) 
        self.convblock2 = StackedBatchNormConv(64, 2)
        self.convblock3 = StackedBatchNormConv(128, 3)
        self.convblock4 = StackedBatchNormConv(128, 3)
        
        self.pooling1 = layers.AveragePooling2D(pool_size=(2,2), padding='same')
        self.pooling2 = layers.AveragePooling2D(pool_size=(2,2), padding='same')
        self.pooling3 = layers.AveragePooling2D(pool_size=(2,2), padding='same')
        self.pooling4 = layers.AveragePooling2D(pool_size=(2,2), padding='same')
        self.ffn = FeedFowardBatchNorm(512, 3)            
        self.reshape = layers.Reshape((-1, self.embedding_dims))        

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, x, training=None):              
        layer = self.convblock1(x) 
        layer = self.pooling1(layer)               
        layer = self.convblock2(layer)
        layer = self.pooling2(layer)
        layer = self.convblock3(layer)
        layer = self.pooling3(layer) 
        layer = self.convblock4(layer)  
        layer = self.pooling4(layer)
        
        layer = self.ffn(layer, training=training) 
        output = self.reshape(layer)       
        
        return output
    
    # serialize model for saving  
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