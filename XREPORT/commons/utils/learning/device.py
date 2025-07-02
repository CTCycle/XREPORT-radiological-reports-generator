from torch import cuda, device
from keras.mixed_precision import set_global_policy

from XREPORT.commons.logger import logger


# [DEVICE SETTINGS]
###############################################################################
class DeviceConfig:
    
    def __init__(self, configuration):
        self.configuration = configuration
            
    #--------------------------------------------------------------------------
    def set_device(self):  
        device_name = 'GPU' if self.configuration.get('use_device_GPU', False) else 'CPU'
        device_id = self.configuration.get('device_ID', 0)
        mixed_precision = self.configuration.get('use_mixed_precision', False)      

        if device_name == 'GPU':
            if not cuda.is_available():
                logger.info('No GPU found. Falling back to CPU')
                dev = device('cpu')
            else:
                dev = device(f'cuda:{device_id}')
                cuda.set_device(dev)  
                logger.info('GPU is set as active device')            
                if mixed_precision:
                    set_global_policy("mixed_float16")
                    logger.info('Mixed precision policy is active during training')                   
        else:
            dev = device('cpu')
            logger.info('CPU is set as active device')  
    
    