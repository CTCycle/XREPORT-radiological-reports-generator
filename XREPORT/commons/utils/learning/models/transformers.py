from torch import compile as torch_compile
from keras import layers, Model, optimizers, activations, ops  
from keras.config import floatx
from keras.saving import register_keras_serializable 

from XREPORT.commons.utils.learning.training.scheduler import WarmUpLRScheduler
from XREPORT.commons.utils.learning.models.encoder import BeitXRayImageEncoder
from XREPORT.commons.utils.learning.models.embeddings import PositionalEmbedding
from XREPORT.commons.utils.learning.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.constants import TOKENIZERS_PATH
from XREPORT.commons.logger import logger


# [ADD NORM LAYER]
###############################################################################
@register_keras_serializable(package='CustomLayers', name='AddNorm')
class AddNorm(layers.Layer):
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
@register_keras_serializable(package='CustomLayers', name='FeedForward')
class FeedForward(layers.Layer):
    def __init__(self, dense_units, dropout, seed, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.dropout_rate = dropout
        self.dense1 = layers.Dense(
            dense_units, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(
            dense_units, activation='relu', kernel_initializer='he_uniform')        
        self.dropout = layers.Dropout(rate=dropout, seed=seed)
        self.seed = seed

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
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
         
# [CLASSIFIER]
###############################################################################
@register_keras_serializable(package='CustomLayers', name='SoftMaxClassifier')
class SoftMaxClassifier(layers.Layer):
    def __init__(self, dense_units, output_size, temperature=1.0, **kwargs):
        super(SoftMaxClassifier, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.output_size = output_size
        self.temperature = temperature
        self.dense1 = layers.Dense(
            dense_units, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(
            output_size, kernel_initializer='he_uniform', dtype=floatx())        

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(SoftMaxClassifier, self).build(input_shape)             

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, x, training=None):
        layer = self.dense1(x)
        layer = activations.relu(layer)
        layer = self.dense2(layer)
        layer = layer/self.temperature 
        output = activations.softmax(layer)         

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(SoftMaxClassifier, self).get_config()
        config.update({'dense_units' : self.dense_units,
                       'output_size' : self.output_size,
                       'temperature' : self.temperature})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [TRANSFORMER ENCODER]
###############################################################################
@register_keras_serializable(package='Encoders', name='TransformerEncoder')
class TransformerEncoder(layers.Layer):
    def __init__(self, embedding_dims, num_heads, seed, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads         
        self.seed = seed                
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embedding_dims, seed=self.seed)
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed)
        # set mask supports to True but mask propagation is handled
        # through the attention layer call        
        self.supports_masking = True   
        self.attention_scores = {} 

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(TransformerEncoder, self).build(input_shape)    

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, mask=None, training=None): 
        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized     
        attention_output, attention_scores = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=None, 
            training=training, return_attention_scores=True)       
        addnorm = self.addnorm1([inputs, attention_output])
        # store self-attention scores for later retrieval
        self.attention_scores['self_attention_scores'] = attention_scores

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm, training=training)
        output = self.addnorm2([addnorm, ffn_out])      

        return output
    
    #--------------------------------------------------------------------------
    def get_attention_scores(self):        
        return self.attention_scores
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({'embedding_dims': self.embedding_dims,
                       'num_heads': self.num_heads,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [TRANSFORMER DECODER]
###############################################################################
@register_keras_serializable(package='Decoders', name='TransformerDecoder')
class TransformerDecoder(layers.Layer):
    def __init__(self, embedding_dims, num_heads, seed, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads  
        self.seed = seed                       
        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embedding_dims, 
            dropout=0.2, seed=self.seed)
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embedding_dims, 
            dropout=0.2, seed=self.seed)        
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed) 
        # set mask supports to True but mask propagation is handled
        # through the attention layer call              
        self.supports_masking = True 
        self.attention_scores = {} 

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(TransformerDecoder, self).build(input_shape)

    # implement transformer decoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, encoder_outputs, mask=None, training=None):        
        causal_mask = self.get_causal_attention_mask(inputs)
        combined_mask = causal_mask

        if mask is not None:
            padding_mask = ops.cast(
                ops.expand_dims(mask, axis=2), dtype='int32')
            combined_mask = ops.minimum(
                ops.cast(ops.expand_dims(mask, axis=1),
                               dtype='int32'), causal_mask)

        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized
        self_masked_MHA, self_attention_scores = self.self_attention(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask, 
            training=training, return_attention_scores=True)       
        addnorm_out1 = self.addnorm1([inputs, self_masked_MHA])
        # store self-attention scores for later retrieval
        self.attention_scores['self_attention_scores'] = self_attention_scores 

        # cross attention using the encoder output as value and key and the output
        # of the addnorm layer as query. The output of this attention layer is then summed
        # to the inputs and normalized
        cross_MHA, cross_attention_scores  = self.cross_attention(
            query=addnorm_out1, value=encoder_outputs, key=encoder_outputs, 
            attention_mask=padding_mask, training=training, return_attention_scores=True)        
        addnorm_out2 = self.addnorm2([addnorm_out1, cross_MHA]) 
        # store cross-attention scores for later retrieval
        self.attention_scores['cross_attention_scores'] = cross_attention_scores 

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn = self.ffn1(addnorm_out2, training=training)
        logits = self.addnorm3([ffn, addnorm_out2])        

        return logits
    
    #--------------------------------------------------------------------------
    def get_attention_scores(self):        
        return self.attention_scores

    # generate causal attention mask   
    #--------------------------------------------------------------------------
    def get_causal_attention_mask(self, inputs):
        batch_size, sequence_length = ops.shape(inputs)[0], ops.shape(inputs)[1]
        i = ops.expand_dims(ops.arange(sequence_length), axis=1)
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype='int32')
        mask = ops.reshape(mask, (1, sequence_length, sequence_length))        
        batch_mask = ops.tile(mask, (batch_size, 1, 1))
        
        return batch_mask
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({'embedding_dims': self.embedding_dims,                       
                       'num_heads': self.num_heads,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     




# [XREP CAPTIONING MODEL]
###############################################################################
class XREPORTModel: 

    def __init__(self, metadata, configuration):         
        self.seed = configuration.get('train_seed', 42)
        self.sequence_length = metadata.get('max_report_size', 200) 
        self.vocabulary_size = metadata.get('vocabulary_size', 200)
        self.img_shape = (224,224,3) 
        
        self.embedding_dims = configuration.get('embedding_dims', 256)
        self.num_heads = configuration.get('num_attention_heads', 3) 
        self.num_encoders = configuration.get('num_encoders', 3) 
        self.num_decoders = configuration.get('num_decoders', 3) 
        self.freeze_img_encoder = configuration.get('freeze_img_encoder', 3)
        self.jit_compile = configuration.get('jit_compile', False)
        self.jit_backend = configuration.get('jit_backend', 'inductor')
        self.has_LR_scheduler = configuration.get('use_scheduler', False)  
        self.post_warm_lr = configuration.get('post_warmup_LR', 0.0001)
        self.warmup_steps = configuration.get('warmup_steps',10000)
        self.temperature = configuration.get('training_temperature', 1.0)
        self.configuration = configuration
        self.metadata = metadata
        
        # initialize the image encoder and the transformers encoders and decoders
        self.img_input = layers.Input(shape=self.img_shape, name='image_input')
        self.seq_input = layers.Input(shape=(self.sequence_length,), name='seq_input')         
        
        self.img_encoder = BeitXRayImageEncoder(
            self.freeze_img_encoder, self.embedding_dims)
        
        self.encoders = [TransformerEncoder(
            self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_encoders)]
        self.decoders = [TransformerDecoder(
            self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_decoders)]        
        self.embeddings = PositionalEmbedding(
            self.vocabulary_size, self.embedding_dims, self.sequence_length) 
        self.classifier = SoftMaxClassifier(
            1024, self.vocabulary_size, self.temperature) 

    #--------------------------------------------------------------------------
    def compile_model(self, model, model_summary=True):
        lr_schedule = self.post_warm_lr
        if self.has_LR_scheduler:            
            post_warmup_LR = self.configuration.get('post_warmup_LR', 40000)   
            warmup_steps = self.configuration.get('warmup_steps', 1000)                        
            lr_schedule = WarmUpLRScheduler(post_warmup_LR, warmup_steps)                  
        
        loss = MaskedSparseCategoricalCrossentropy()  
        metric = [MaskedAccuracy()]
        opt = optimizers.AdamW(learning_rate=lr_schedule)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)                 
  
        model.summary(expand_nested=True) if model_summary else None
        if self.jit_compile:
            model = torch_compile(model, backend=self.jit_backend, mode='default')

        return model                

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
        # encode images and extract their features using the convolutional 
        # image encoder or a selected pretrained model
        image_features = self.img_encoder(self.img_input)      
        embeddings = self.embeddings(self.seq_input) 
        padding_mask = self.embeddings.compute_mask(self.seq_input)         
                
        encoder_output = image_features
        decoder_output = embeddings    
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=False)
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, encoder_output, 
                                     training=False, mask=padding_mask)

        # apply the softmax classifier layer
        output = self.classifier(decoder_output)    

        # wrap the model and compile it with AdamW optimizer
        model = Model(inputs=[self.img_input, self.seq_input], outputs=output)    
        model = self.compile_model(model, model_summary=model_summary)   

        return model
       



