import keras
from transformers import AutoImageProcessor, AutoModel

from XREPORT.commons.constants import CONFIG, ENCODERS_PATH
from XREPORT.commons.logger import logger


# [PRETRAINED IMAGE ENCODER]
###############################################################################
@keras.utils.register_keras_serializable(package='Encoders', name='BeitXRayImageEncoder')
class BeitXRayImageEncoder(keras.layers.Layer):
    def __init__(self, freeze_layers=False, embedding_dims=256, **kwargs):
        super(BeitXRayImageEncoder, self).__init__(**kwargs)   
        self.encoder_name = 'microsoft/beit-base-patch16-224'        
        self.freeze_layers = freeze_layers
        self.embedding_dims = embedding_dims        

        self.model = AutoModel.from_pretrained(self.encoder_name, cache_dir=ENCODERS_PATH)
        if self.freeze_layers is True:            
            for param in self.model.parameters():
                param.requires_grad = False  

        self.processor = AutoImageProcessor.from_pretrained(
            self.encoder_name, cache_dir=ENCODERS_PATH, use_fast=True)  

        self.dense = keras.layers.Dense(self.embedding_dims)   

    # call method
    #--------------------------------------------------------------------------
    def call(self, inputs, **kwargs): 
        inputs = keras.ops.transpose(inputs, axes=(0, 3, 1, 2))             
        outputs = self.model(inputs, **kwargs)
        output = outputs.last_hidden_state
        output = self.dense(output)
                
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(BeitXRayImageEncoder, self).get_config()
        config.update({'freeze_layers': self.freeze_layers,
                       'embedding_dims': self.embedding_dims})

        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)



