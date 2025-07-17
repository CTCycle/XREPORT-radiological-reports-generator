from keras.config import floatx
from keras.ops import cast, cond
from keras.optimizers.schedules import LearningRateSchedule 
from keras.saving import register_keras_serializable


from XREPORT.app.logger import logger
           
# [LEARNING RATE SCHEDULER]
###############################################################################
@register_keras_serializable(package='WarmUpLRScheduler')
class WarmUpLRScheduler(LearningRateSchedule):
    def __init__(self, post_warmup_lr, warmup_steps, **kwargs):
        super(WarmUpLRScheduler, self).__init__(**kwargs)
        self.post_warmup_lr = post_warmup_lr
        self.warmup_steps = warmup_steps

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def __call__(self, step):
        global_step = cast(step, floatx())
        warmup_steps = cast(self.warmup_steps, floatx())
        # Linear warmup: gradually increase lr from 0 to post_warmup_lr
        warmup_lr = self.post_warmup_lr * (global_step / warmup_steps)        
        # Inverse square root decay after warmup:
        # At global_step == warmup_steps, decayed_lr equals post_warmup_lr.
        # For global_step > warmup_steps, the learning rate decays as sqrt(warmup_steps/global_step)
        decayed_lr = self.post_warmup_lr * (global_step / warmup_steps) ** (-0.5)
        
        # Use keras.ops.cond to select the appropriate phase
        return cond(global_step < warmup_steps,
                              lambda: warmup_lr,
                              lambda: decayed_lr)
    
    # custom configuration
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
    
      
