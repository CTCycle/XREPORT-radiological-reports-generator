from keras.config import floatx
from keras.ops import cast, cond
from keras.optimizers.schedules import LearningRateSchedule
from keras.saving import register_keras_serializable


# [LEARNING RATE SCHEDULER]
###############################################################################
@register_keras_serializable(package="WarmUpLRScheduler")
class WarmUpLRScheduler(LearningRateSchedule):
    def __init__(self, post_warmup_LR, warmup_steps, **kwargs):
        super(WarmUpLRScheduler, self).__init__(**kwargs)
        self.post_warmup_LR = post_warmup_LR
        self.warmup_steps = warmup_steps

    # implement encoder through call method
    # -------------------------------------------------------------------------
    def __call__(self, step):
        global_step = cast(step, floatx())
        warmup_steps = cast(self.warmup_steps, floatx())
        # Linear warmup: gradually increase lr from 0 to post_warmup_LR
        warmup_LR = self.post_warmup_LR * (global_step / warmup_steps)
        # Inverse square root decay after warmup:
        # At global_step == warmup_steps, decayed_LR equals post_warmup_LR.
        # For global_step > warmup_steps, the learning rate decays as sqrt(warmup_steps/global_step)
        decayed_LR = self.post_warmup_LR * (global_step / warmup_steps) ** (-0.5)

        # Use keras.ops.cond to select the appropriate phase
        return cond(global_step < warmup_steps, lambda: warmup_LR, lambda: decayed_LR)

    # custom configuration
    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = {
            "post_warmup_LR": self.post_warmup_LR,
            "warmup_steps": self.warmup_steps,
        }

        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
