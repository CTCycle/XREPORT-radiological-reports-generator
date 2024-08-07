import torch
import keras
import tensorflow as tf

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
           
# [LEARNING RATE SCHEDULER]
###############################################################################
@keras.utils.register_keras_serializable(package='LRScheduler')
class LRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_lr, warmup_steps, **kwargs):
        super(LRScheduler).__init__(**kwargs)
        self.post_warmup_lr = post_warmup_lr
        self.warmup_steps = warmup_steps

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def __call__(self, step):
        global_step = keras.ops.cast(step, torch.float32)
        warmup_steps = keras.ops.cast(self.warmup_steps, torch.float32)
        warmup_progress = global_step/warmup_steps
        warmup_learning_rate = self.post_warmup_lr * warmup_progress

        return keras.ops.cond(global_step < warmup_steps, lambda: warmup_learning_rate,
                              lambda: self.post_warmup_lr)
    
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
    
      
