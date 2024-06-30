import tensorflow as tf
from tensorflow import keras
from keras import layers    

from XREPORT.commons.utils.models.embeddings import PositionalEmbedding
from XREPORT.commons.pathfinder import CHECKPOINT_PATH
from XREPORT.commons.configurations import (BATCH_SIZE, IMG_SHAPE, EMBEDDING_DIMS,
                                            MAX_CAPTION_SIZE, NUM_HEADS, SEED)


# [ADD NORM LAYER]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='CustomLayers', name='AddNorm')
class AddNorm(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization()

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, x1, x2, training=None):

        x_add = self.add([x1, x2])
        x_norm = self.layernorm(x_add)

        return x_norm
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(AddNorm, self).get_config()
        config.update({})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
# [FEED FORWARD]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='CustomLayers', name='FeedForward')
class FeedForward(keras.layers.Layer):
    def __init__(self, dense_units, dropout, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.dropout = dropout
        self.dense1 = layers.Dense(dense_units, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(dense_units, activation='relu', kernel_initializer='he_uniform')        
        self.dropout = layers.Dropout(dropout, SEED)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, x, training=None):

        x = self.dense1(x)
        x = self.dense2(x)  
        output = self.dropout(x, training=training) 

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeedForward, self).get_config()
        config.update({'dense_units' : self.dense_units,
                       'dropout' : self.dropout,
                       'seed' : SEED})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
         
# [CLASSIFIER]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='CustomLayers', name='SoftMaxClassifier')
class SoftMaxClassifier(keras.layers.Layer):
    def __init__(self, dense_units, output_size, **kwargs):
        super(SoftMaxClassifier, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.output_size = output_size
        self.dense1 = layers.Dense(dense_units, activation='relu', 
                                   kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(output_size, activation='softmax', 
                                   kernel_initializer='he_uniform', dtype=tf.float32)       
        

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, x, training=None):

        x = self.dense1(x)
        output = self.dense2(x)          

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(SoftMaxClassifier, self).get_config()
        config.update({'dense_units' : self.dense_units,
                       'output_size' : self.output_size})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# [TRANSFORMER ENCODER]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='Encoders', name='TransformerEncoderBlock')
class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)          
        self.attention = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBEDDING_DIMS)
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.ffn1 = FeedForward(512, 0.2)        

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):        

        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized     
        attention_output = self.attention(query=inputs, value=inputs, key=inputs,
                                          attention_mask=None, training=training)
         
        addnorm = self.addnorm1([inputs, attention_output])

        # feed forward network with ReLU activation to firther process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm)
        output = self.addnorm2([addnorm, ffn_out])      

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({'embedding_dims': EMBEDDING_DIMS,
                       'num_heads': NUM_HEADS,
                       'seed': SEED})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [TRANSFORMER DECODER]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='Decoders', name='TransformerDecoderBlock')
class TransformerDecoderBlock(keras.layers.Layer):
    def __init__(self, vocab_size, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.vocab_size = vocab_size
            
        self.posembedding = PositionalEmbedding(MAX_CAPTION_SIZE, vocab_size, EMBEDDING_DIMS, mask_zero=True)          
        self.self_attention = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBEDDING_DIMS, dropout=0.2)
        self.cross_attention = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBEDDING_DIMS, dropout=0.2)        
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()
        self.ffn1 = FeedForward(EMBEDDING_DIMS, 0.2)            
        self.supports_masking = True 

    # implement transformer decoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, encoder_outputs, training=True, mask=None):

        
        inputs = self.posembedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        padding_mask = None
        combined_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask) 

        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized
        self_masked_MHA = self.self_attention(query=inputs, value=inputs, key=inputs,
                                              attention_mask=combined_mask, training=training)        
        addnorm_out1 = self.addnorm1(inputs + self_masked_MHA) 

        # cross attention using the encoder output as value and key and the output
        # of the addnorm layer as query. The output of this attention layer is then summed
        # to the inputs and normalized
        cross_MHA = self.cross_attention(query=addnorm_out1, value=encoder_outputs,
                                                 key=encoder_outputs, attention_mask=padding_mask,
                                                 training=training)        
        addnorm_out2 = self.addnorm2(addnorm_out1 + cross_MHA) 

        # feed forward network with ReLU activation to firther process the output
        # addition and layer normalization of inputs and outputs
        ffn = self.ffn1(addnorm_out2)
        addnorm_out3 = self.addnorm2(ffn + addnorm_out2) 
           
        output = self.outmax(ffn_out)

        return preds

    # generate causal attention mask   
    #--------------------------------------------------------------------------
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
                          axis=0)
        
        return tf.tile(mask, mult) 
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerDecoderBlock, self).get_config()
        config.update({'sequence_length': self.sequence_length,
                       'vocab_size': self.vocab_size,
                       'embedding_dims': self.embedding_dims,                       
                       'num_heads': self.num_heads,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     

