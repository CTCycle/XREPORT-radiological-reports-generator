import torch
import keras
from keras import layers    

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [ADD NORM LAYER]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='AddNorm')
class AddNorm(keras.layers.Layer):
    def __init__(self, epsilon=10e-5, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization(epsilon=self.epsilon)    

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(AddNorm, self).build(input_shape)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------        
    def call(self, inputs):
        x1, x2 = inputs
        x_add = self.add([x1, x2])
        x_norm = self.layernorm(x_add)

        return x_norm
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(AddNorm, self).get_config()
        config.update({'epsilon' : self.epsilon})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
# [FEED FORWARD]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='FeedForward')
class FeedForward(keras.layers.Layer):
    def __init__(self, dense_units, dropout, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.dropout_rate = dropout
        self.dense1 = layers.Dense(dense_units, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(dense_units, activation='relu', kernel_initializer='he_uniform')        
        self.dropout = layers.Dropout(rate=dropout, seed=CONFIG["SEED"])

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(FeedForward, self).build(input_shape)

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
                       'dropout_rate' : self.dropout_rate,
                       'seed' : CONFIG["SEED"]})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
         
# [CLASSIFIER]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='SoftMaxClassifier')
class SoftMaxClassifier(keras.layers.Layer):
    def __init__(self, dense_units, output_size, **kwargs):
        super(SoftMaxClassifier, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.output_size = output_size
        self.dense1 = layers.Dense(dense_units, activation='relu', 
                                   kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(output_size, activation='softmax', 
                                   kernel_initializer='he_uniform', dtype=torch.float32)

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(SoftMaxClassifier, self).build(input_shape)     
        

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
###############################################################################
@keras.utils.register_keras_serializable(package='Encoders', name='TransformerEncoder')
class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads                 
        self.attention = layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                   key_dim=self.embedding_dims)
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2)    

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(TransformerEncoder, self).build(input_shape)    

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None):        

        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized     
        attention_output = self.attention(query=inputs, value=inputs, key=inputs,
                                          attention_mask=None, training=training)
         
        addnorm = self.addnorm1([inputs, attention_output])

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm, training=training)
        output = self.addnorm2([addnorm, ffn_out])      

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({'embedding_dims': self.embedding_dims,
                       'num_heads': self.num_heads})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [TRANSFORMER DECODER]
###############################################################################
@keras.utils.register_keras_serializable(package='Decoders', name='TransformerDecoder')
class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads                         
        self.self_attention = layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                        key_dim=self.embedding_dims, 
                                                        dropout=0.2)
        self.cross_attention = layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                         key_dim=self.embedding_dims, 
                                                         dropout=0.2)        
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2)            
        self.supports_masking = True 

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(TransformerDecoder, self).build(input_shape)

    # implement transformer decoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, encoder_outputs, training=None, mask=None):        
        
        causal_mask = self.get_causal_attention_mask(inputs)
        combined_mask = causal_mask

        if mask is not None:
            padding_mask = keras.ops.cast(keras.ops.expand_dims(mask, axis=2), dtype=torch.int32)
            combined_mask = keras.ops.minimum(keras.ops.cast(keras.ops.expand_dims(mask, axis=1), 
                                                             dtype=torch.int32), causal_mask)

        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized
        self_masked_MHA = self.self_attention(query=inputs, value=inputs, key=inputs,
                                              attention_mask=combined_mask, training=training)        
        addnorm_out1 = self.addnorm1([inputs, self_masked_MHA]) 

        # cross attention using the encoder output as value and key and the output
        # of the addnorm layer as query. The output of this attention layer is then summed
        # to the inputs and normalized
        cross_MHA = self.cross_attention(query=addnorm_out1, value=encoder_outputs,
                                         key=encoder_outputs, attention_mask=padding_mask,
                                         training=training)        
        addnorm_out2 = self.addnorm2([addnorm_out1, cross_MHA]) 

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn = self.ffn1(addnorm_out2, training=training)
        logits = self.addnorm3([ffn, addnorm_out2])        

        return logits

    # generate causal attention mask   
    #--------------------------------------------------------------------------
    def get_causal_attention_mask(self, inputs):

        batch_size, sequence_length = keras.ops.shape(inputs)[0], keras.ops.shape(inputs)[1]
        i = keras.ops.expand_dims(keras.ops.arange(sequence_length), axis=1)
        j = keras.ops.arange(sequence_length)
        mask = keras.ops.cast(i >= j, dtype=torch.int32)
        mask = keras.ops.reshape(mask, (1, sequence_length, sequence_length))        
        batch_mask = keras.ops.tile(mask, (batch_size, 1, 1))
        
        return batch_mask
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({'embedding_dims': self.embedding_dims,                       
                       'num_heads': self.num_heads})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     

