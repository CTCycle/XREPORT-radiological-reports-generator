import os
import re
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras import layers


from XREPORT.commons.utils.models.transformers import TransformerEncoderBlock, TransformerDecoderBlock
from XREPORT.commons.utils.models.image_encoding import ImageEncoder
from XREPORT.commons.constants import CONFIG

    

 

# [XREP CAPTIONING MODEL]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='Models', name='XREPCaptioningModel')
class XREPCaptioningModel(keras.Model):    
    def __init__(self, vocab_size, **kwargs):   
        super(XREPCaptioningModel, self).__init__(**kwargs)        
        self.learning_rate = CONFIG["training"]["LEARNING_RATE"] 
        self.xla_state = CONFIG["training"]["XLA_STATE"]
        self.num_encoders = CONFIG["model"]["NUM_ENCODERS"]
        self.num_decoders = CONFIG["model"]["NUM_DECODERS"]
        
        # add loss and metrics tracker to average value for all batches
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.acc_tracker = keras.metrics.Mean(name='accuracy')        
        self.vocab_size = vocab_size 
          
        # initialize custom blocks and layers            
        self.image_encoder = ImageEncoder()        
        self.encoders = [TransformerEncoderBlock() for i in range(self.num_encoders)]        
        self.decoders = [TransformerDecoderBlock(self.vocab_size) for i in range(self.num_decoders)] 

    # calculate loss
    #--------------------------------------------------------------------------
    def calculate_loss(self, y_true, y_pred, mask):               
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss)/(tf.reduce_sum(mask) + keras.backend.epsilon())

        return loss
    
    # calculate accuracy
    #--------------------------------------------------------------------------
    def calculate_accuracy(self, y_true, y_pred, mask): 
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred_argmax = tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.float32)
        accuracy = tf.equal(y_true, y_pred_argmax)
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        accuracy = tf.reduce_sum(accuracy) / (tf.reduce_sum(mask) + keras.backend.epsilon())

        return accuracy

    
    
    # define train step
    #--------------------------------------------------------------------------
    def train_step(self, batch_data):
        x_data, y_data = batch_data
        batch_img, batch_seq = x_data
        img_embed = self.image_encoder(batch_img)
        with tf.GradientTape() as tape:

            batch_seq_inp = batch_seq[:, :-1]
            batch_seq_true = batch_seq[:, 1:]            
            encoder_out = img_embed
            for encoder in self.encoders:
                encoder_out = encoder(encoder_out, training=training)
            batch_seq_pred = batch_seq_inp
            for decoder in self.decoders:
                batch_seq_pred = decoder(batch_seq_pred, encoder_out, training=training, mask=mask)
            loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)


        encoder_vars = [var for encoder in self.encoders for var in encoder.trainable_variables]
        train_vars = encoder_vars + [var for decoder in self.decoders for var in decoder.trainable_variables]
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {'loss': self.loss_tracker.result(), 
                'accuracy': self.acc_tracker.result()}

    # define test step
    #--------------------------------------------------------------------------
    def test_step(self, batch_data):
        x_data, y_data = batch_data
        batch_img, batch_seq = x_data
        img_embed = self.image_encoder(batch_img)
        loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=False)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {'loss': self.loss_tracker.result(), 
                'accuracy': self.acc_tracker.result()}    
 
    # implement captioning model through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=True):

        images, sequences = inputs
        mask = tf.math.not_equal(sequences, 0)
        image_features = self.image_encoder(images)
        encoder_out = image_features
        batch_seq_pred = sequences
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, training=training)
        decoder_out = batch_seq_pred
        for decoder in self.decoders:
            decoder_out = decoder(decoder_out, encoder_out, training=training, mask=mask)

        return decoder_out  
    
    # print summary
    #--------------------------------------------------------------------------
    def get_model(self):
        image_input = layers.Input(shape=self.picture_shape)    
        seq_input = layers.Input(shape=(self.sequence_length, ))
        model = Model(inputs=[image_input, seq_input], outputs = self.call([image_input, seq_input], 
                      training=False)) 
        
        return model       

    # compile the model
    #--------------------------------------------------------------------------
    def compile(self):
        lr_schedule = LRScheduler(self.learning_rate, warmup_steps=10)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, 
                                                          reduction=keras.losses.Reduction.NONE)  
        metric = keras.metrics.SparseCategoricalAccuracy()  
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)          
        super(XREPCaptioningModel, self).compile(optimizer=opt, loss=loss, metrics=metric, 
                                                 run_eagerly=False, jit_compile=self.XLA_state)   
        
    # track metrics and losses  
    #--------------------------------------------------------------------------
    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]     
 
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(XREPCaptioningModel, self).get_config()
        config.update({'picture_shape': self.picture_shape,
                       'sequence_length': self.sequence_length,
                       'vocab_size': self.vocab_size,
                       'embedding_dims': self.embedding_dims,                       
                       'kernel_size': self.kernel_size,
                       'num_heads': self.num_heads,
                       'learning_rate' : self.learning_rate,
                       'XLA_state' : self.XLA_state,                 
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    



