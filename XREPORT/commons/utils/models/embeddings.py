import torch
import keras
import tensorflow as tf
from keras import activations, layers 

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
      

# [POSITIONAL EMBEDDING]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='PositionalEmbedding')
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, mask_zero=True, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.embedding_dims = CONFIG["model"]["EMBEDDING_DIMS"]
        self.sequence_length = CONFIG["dataset"]["MAX_REPORT_SIZE"]
        self.vocab_size = vocab_size
        self.mask_zero = mask_zero
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_dims, mask_zero=mask_zero)
        self.position_embeddings = layers.Embedding(input_dim=self.sequence_length, output_dim=self.embedding_dims)
        self.embedding_scale = keras.ops.sqrt(keras.ops.cast(self.embedding_dims, torch.float32))       
    
    # implement positional embedding through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs):
        length = keras.ops.shape(inputs)[-1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        positions = keras.ops.cast(positions, dtype=inputs.dtype)        
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens *= self.embedding_scale        
        embedded_positions = self.position_embeddings(positions)        
        full_embedding = embedded_tokens + embedded_positions
        
        if self.mask_zero:
            mask = keras.ops.not_equal(inputs, 0)
            mask = keras.ops.expand_dims(keras.ops.cast(mask, torch.float32), axis=-1)
            full_embedding *= mask

        return full_embedding
    
    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):
        mask = keras.ops.not_equal(inputs, 0)        
        
        return mask
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({'sequence_length': self.sequence_length,
                       'vocab_size': self.vocab_size,
                       'embedding_dims': self.embedding_dims,                       
                       'mask_zero': self.mask_zero})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

