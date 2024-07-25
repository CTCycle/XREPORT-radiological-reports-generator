import tensorflow as tf
from tensorflow import keras
import keras.backend as K

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [LOSS FUNCTION]
###############################################################################
class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    
    def __init__(self, name='MaskedSparseCategoricalCrossentropy', **kwargs):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
    #--------------------------------------------------------------------------    
    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.math.not_equal(y_true, 0)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss)/(tf.reduce_sum(mask) + keras.backend.epsilon())

        return loss
    
    #--------------------------------------------------------------------------    
    def get_config(self):
        base_config = super(MaskedSparseCategoricalCrossentropy, self).get_config()
        return {**base_config, 'name': self.name}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
# [METRICS]
###############################################################################
class MaskedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='MaskedAccuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    #--------------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred_argmax = tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.float32)
        accuracy = tf.equal(y_true, y_pred_argmax)        
        # Create a mask to ignore padding (assuming padding value is 0)
        mask = tf.math.not_equal(y_true, 0)        
        # Apply the mask to the accuracy
        accuracy = tf.math.logical_and(mask, accuracy)        
        # Cast the boolean values to float32
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            accuracy = tf.multiply(accuracy, sample_weight)
            mask = tf.multiply(mask, sample_weight)
        
        # Update the state variables
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.reduce_sum(mask))
    
    #--------------------------------------------------------------------------
    def result(self):
        return self.total / (self.count + K.epsilon())
    
    #--------------------------------------------------------------------------
    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)

    #--------------------------------------------------------------------------
    def get_config(self):
        base_config = super(MaskedAccuracy, self).get_config()
        return {**base_config, 'name': self.name}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)







