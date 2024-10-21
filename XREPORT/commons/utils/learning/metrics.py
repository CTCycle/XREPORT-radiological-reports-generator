import torch
import keras

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [LOSS FUNCTION]
###############################################################################
class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    
    def __init__(self, name='MaskedSparseCategoricalCrossentropy', **kwargs):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                               reduction=None)
        
    #--------------------------------------------------------------------------    
    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = keras.ops.not_equal(y_true, 0)        
        mask = keras.ops.cast(mask, dtype=loss.dtype)        
        loss *= mask
        loss = keras.ops.sum(loss)/(keras.ops.sum(mask) + keras.backend.epsilon())

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
class MaskedAccuracy(keras.metrics.Metric):

    def __init__(self, name='MaskedAccuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    #--------------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = keras.ops.cast(y_true, dtype=torch.float32)
        y_pred_argmax = keras.ops.cast(keras.ops.argmax(y_pred, axis=2), dtype=torch.float32)
        accuracy = keras.ops.equal(y_true, y_pred_argmax)        
        # Create a mask to ignore padding (assuming padding value is 0)
        mask = keras.ops.not_equal(y_true, 0)        
        # Apply the mask to the accuracy
        accuracy = keras.ops.logical_and(mask, accuracy)        
        # Cast the boolean values to float32
        accuracy = keras.ops.cast(accuracy, dtype=torch.float32)
        mask = keras.ops.cast(mask, dtype=torch.float32)
        
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, dtype=torch.float32)
            accuracy = keras.ops.multiply(accuracy, sample_weight)
            mask = keras.ops.multiply(mask, sample_weight)
        
        # Update the state variables
        self.total.assign_add(keras.ops.sum(accuracy))
        self.count.assign_add(keras.ops.sum(mask))
    
    #--------------------------------------------------------------------------
    def result(self):
        return self.total / (self.count + keras.backend.epsilon())
    
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







