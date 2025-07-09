import torch
from keras.mixed_precision import set_global_policy

from XREPORT.app.src.logger import logger


# [DEVICE SETTINGS]
###############################################################################
class DeviceConfig:
    
    def __init__(self, configuration):
        self.configuration = configuration
            
    #--------------------------------------------------------------------------
    def set_device(self):  
        use_gpu = self.configuration.get('use_device_GPU', False)
        device_name = 'cuda' if use_gpu else 'cpu'
        device_id = self.configuration.get('device_ID', 0)
        mixed_precision = self.configuration.get('use_mixed_precision', False)      

        if device_name == 'cuda' and torch.cuda.is_available():
            device_id = self.configuration.get('device_ID', 0)
            dev = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(dev)
            logger.info(f'GPU (cuda:{device_id}) is set as the active device.')                   
            if mixed_precision:
                set_global_policy("mixed_float16")
                logger.info('Mixed precision policy is active during training')                   
        else:            
            if device_name == 'cuda':
                logger.info('No GPU found. Falling back to CPU.')
            dev = torch.device('cpu')
            logger.info('CPU is set as the active device.')

  
    
    