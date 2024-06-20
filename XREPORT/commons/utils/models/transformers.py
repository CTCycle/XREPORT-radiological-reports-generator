import tensorflow as tf
from tensorflow import keras
from keras import layers    

    

    


# [TRANSFORMER ENCODER]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='Encoders', name='TransformerEncoderBlock')
class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads, seed, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims       
        self.num_heads = num_heads  
        self.seed = seed       
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dense1 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.dense3 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dense4 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dropout1 = layers.Dropout(0.2, seed=seed)
        self.dropout2 = layers.Dropout(0.3, seed=seed)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):        
        inputs = self.layernorm1(inputs)
        inputs = self.dense1(inputs)
        inputs = self.dropout1(inputs)  
        inputs = self.dense2(inputs)            
        attention_output = self.attention(query=inputs, value=inputs, key=inputs,
                                          attention_mask=None, training=training)        
        layernorm = self.layernorm2(inputs + attention_output)
        layer = self.dense3(layernorm)
        layer = self.dropout2(layer)
        output = self.dense4(layer)        

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({'embedding_dims': self.embedding_dims,
                       'num_heads': self.num_heads,
                       'seed': self.seed})
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
    def __init__(self, sequence_length, vocab_size, embedding_dims, num_heads, seed, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims              
        self.num_heads = num_heads
        self.seed = seed        
        self.posembedding = PositionalEmbedding(sequence_length, vocab_size, embedding_dims, mask_zero=True)          
        self.MHA_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims, dropout=0.2)
        self.MHA_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims, dropout=0.2)
        self.FFN_1 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.FFN_2 = layers.Dense(self.embedding_dims, activation='relu', kernel_initializer='he_uniform')
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()
        self.dense = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')         
        self.outmax = layers.Dense(self.vocab_size, activation='softmax')
        self.dropout1 = layers.Dropout(0.2, seed=seed)
        self.dropout2 = layers.Dropout(0.3, seed=seed) 
        self.dropout3 = layers.Dropout(0.3, seed=seed)
        self.supports_masking = True 

    # implement transformer decoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, encoder_outputs, training=True, mask=None):
        inputs = tf.cast(inputs, tf.int32)
        inputs = self.posembedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)        
        padding_mask = None
        combined_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)           
        attention_output1 = self.MHA_1(query=inputs, value=inputs, key=inputs,
                                       attention_mask=combined_mask, training=training)
        output1 = self.layernorm1(inputs + attention_output1)                       
        attention_output2 = self.MHA_2(query=output1, value=encoder_outputs,
                                       key=encoder_outputs, attention_mask=padding_mask,
                                       training=training)
        output2 = self.layernorm2(output1 + attention_output2)
        ffn_out = self.FFN_1(output2)
        ffn_out = self.dropout1(ffn_out, training=training)
        ffn_out = self.FFN_2(ffn_out)
        ffn_out = self.layernorm3(ffn_out + output2, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)         
        ffn_out = self.dense(ffn_out)   
        ffn_out = self.dropout3(ffn_out, training=training)     
        preds = self.outmax(ffn_out)

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
     

