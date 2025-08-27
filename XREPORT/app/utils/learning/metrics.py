from keras.losses import Loss, SparseCategoricalCrossentropy
from keras.metrics import Metric
from keras import ops, backend
from keras.config import floatx



# [LOSS FUNCTION]
###############################################################################
class MaskedSparseCategoricalCrossentropy(Loss):
    
    def __init__(self, name='MaskedSparseCategoricalCrossentropy', **kwargs):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.loss = SparseCategoricalCrossentropy(
            from_logits=False, reduction=None)
        
    #--------------------------------------------------------------------------    
    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = ops.not_equal(y_true, 0)        
        mask = ops.cast(mask, dtype=loss.dtype)        
        loss *= mask
        loss = ops.sum(loss)/(ops.sum(mask) + backend.epsilon())

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
class MaskedAccuracy(Metric):

    def __init__(self, name='MaskedAccuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    #--------------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = ops.cast(y_true, dtype=floatx())
        y_pred_argmax = ops.cast(ops.argmax(y_pred, axis=2), dtype=floatx())
        accuracy = ops.equal(y_true, y_pred_argmax)        
        # Create a mask to ignore padding (assuming padding value is 0)
        mask = ops.not_equal(y_true, 0)        
        # Apply the mask to the accuracy
        accuracy = ops.logical_and(mask, accuracy)        
        # Cast the boolean values to float32
        accuracy = ops.cast(accuracy, dtype=floatx())
        mask = ops.cast(mask, dtype=floatx())
        
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=floatx())
            accuracy = ops.multiply(accuracy, sample_weight)
            mask = ops.multiply(mask, sample_weight)
        
        # Update the state variables
        self.total.assign_add(ops.sum(accuracy))
        self.count.assign_add(ops.sum(mask))
    
    #--------------------------------------------------------------------------
    def result(self):
        return self.total / (self.count + backend.epsilon())
    
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







