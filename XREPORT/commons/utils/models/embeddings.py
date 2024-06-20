import tensorflow as tf
from tensorflow import keras
from keras import layers


      

# [POSITIONAL EMBEDDING]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='CustomLayers', name='PositionalEmbedding')
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embedding_dims, mask_zero=True, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size 
        self.embedding_dim = embedding_dims        
        self.mask_zero = mask_zero        
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dims)                        
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embedding_dims)        
        self.embedding_scale = tf.math.sqrt(tf.cast(embedding_dims, tf.float32))        
    
    # implement positional embedding through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1) 
        positions = tf.cast(positions, dtype=inputs.dtype)       
        embedded_tokens = self.token_embeddings(inputs)  
        embedded_tokens = embedded_tokens * self.embedding_scale
        embedded_positions = self.position_embeddings(positions)
        full_embedding = embedded_tokens + embedded_positions        
        if self.mask_zero==True:
            mask = tf.math.not_equal(inputs, 0)
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)                              
            full_embedding = full_embedding * mask            

        return full_embedding
    
    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({'sequence_length': self.sequence_length,
                       'vocab_size': self.vocab_size,
                       'embedding_dims': self.embedding_dim,
                       'bio_path' : self.bio_path,
                       'mask_zero': self.mask_zero})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

