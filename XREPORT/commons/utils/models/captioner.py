import os
import re
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras import layers

from XREPORT.commons.utils.models.scheduler import LRScheduler
from XREPORT.commons.utils.models.transformers import TransformerEncoderBlock, TransformerDecoderBlock
from XREPORT.commons.utils.models.image_encoding import ImageEncoder
from XREPORT.commons.constants import CONFIG


# [XREP CAPTIONING MODEL]
#------------------------------------------------------------------------------
class XREPORTModel: 

    def __init__(self, vocab_size): 
        self.vocab_size = vocab_size
        self.img_shape = CONFIG["model"]["IMG_SHAPE"] 
        self.sequence_length = CONFIG["dataset"]["MAX_CAPTION_SIZE"]        
        self.num_layers = CONFIG["model"]["NUM_LAYERS"]        
        self.learning_rate = CONFIG["training"]["LEARNING_RATE"] 
        self.xla_state = CONFIG["training"]["XLA_STATE"]            
                

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, summary=True): 

        image_input = layers.Input(shape=self.img_shape, name='image_input')
        seq_input = layers.Input(shape=(self.sequence_length,), name='seq_input')

        image_encoder = ImageEncoder()
        encoders = [TransformerEncoderBlock() for _ in range(self.num_layers)]
        decoders = [TransformerDecoderBlock(self.vocab_size) for _ in range(self.num_layers)]

        mask = tf.math.not_equal(seq_input, 0)
        image_features = image_encoder(image_input)
        
        encoder_output = image_features
        decoder_output = seq_input
        for encoder, decoder in zip(encoders, decoders):
            encoder_output = encoder(encoder_output, training=False)
            decoder_output = decoder(decoder_output, encoder_output, training=False, mask=mask)

        output = layers.Softmax(decoder_output)    

        model = Model(inputs=[image_input, seq_input], outputs=output)       
    
        lr_schedule = LRScheduler(self.learning_rate, warmup_steps=10)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, 
                                                          reduction=keras.losses.Reduction.NONE)  
        metric = keras.metrics.SparseCategoricalAccuracy()  
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, 
                      jit_compile=self.xla_state)         
        if summary:
            model.summary(expand_nested=True)

        return model
       



