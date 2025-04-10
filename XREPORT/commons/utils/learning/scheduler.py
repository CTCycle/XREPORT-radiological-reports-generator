import torch
import keras

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
           
# [LEARNING RATE SCHEDULER]
###############################################################################
@keras.saving.register_keras_serializable(package='WarmUpLRScheduler')
class WarmUpLRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_lr, warmup_steps, **kwargs):
        super(WarmUpLRScheduler, self).__init__(**kwargs)
        self.post_warmup_lr = post_warmup_lr
        self.warmup_steps = warmup_steps

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def __call__(self, step):
        global_step = keras.ops.cast(step, keras.config.floatx())
        warmup_steps = keras.ops.cast(self.warmup_steps, keras.config.floatx())
        # Linear warmup: gradually increase lr from 0 to post_warmup_lr
        warmup_lr = self.post_warmup_lr * (global_step / warmup_steps)        
        # Inverse square root decay after warmup:
        # At global_step == warmup_steps, decayed_lr equals post_warmup_lr.
        # For global_step > warmup_steps, the learning rate decays as sqrt(warmup_steps/global_step)
        decayed_lr = self.post_warmup_lr * (global_step / warmup_steps) ** (-0.5)
        
        # Use keras.ops.cond to select the appropriate phase
        return keras.ops.cond(global_step < warmup_steps,
                              lambda: warmup_lr,
                              lambda: decayed_lr)
    
    # custom configurations
    #--------------------------------------------------------------------------
    def get_config(self):
        
        config = {'post_warmup_lr': self.post_warmup_lr,
                  'warmup_steps': self.warmup_steps}
        
        return config        
    
    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
      
