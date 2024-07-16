import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from XREPORT.commons.utils.models.callbacks import RealTimeHistory, LoggingCallback
from XREPORT.commons.utils.dataloader.serializer import ModelSerializer
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


           
# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
#------------------------------------------------------------------------------
class ModelTraining:    
       
    def __init__(self):                            
        np.random.seed(CONFIG["SEED"])
        tf.random.set_seed(CONFIG["SEED"])         
        self.available_devices = tf.config.list_physical_devices()               
        logger.info('The current devices are available:\n')        
        for dev in self.available_devices:            
            logger.info(dev)

    # set device
    #--------------------------------------------------------------------------
    def set_device(self):
       
        if CONFIG["training"]["ML_DEVICE"] == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                logger.info('No GPU found. Falling back to CPU')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if CONFIG["training"]["MIXED_PRECISION"]:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
                os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'                 
                logger.info('GPU is set as active device')
                   
        elif CONFIG["training"]["ML_DEVICE"] == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            logger.info('CPU is set as active device')    

    #--------------------------------------------------------------------------
    def train_model(self, model : tf.keras.Model, train_data, 
                    validation_data, current_checkpoint_path):

        # initialize the real time history callback    
        RTH_callback = RealTimeHistory(current_checkpoint_path)
        logger_callback = LoggingCallback()
        callbacks_list = [RTH_callback, logger_callback]

        # initialize tensorboard if requested    
        if CONFIG["training"]["USE_TENSORBOARD"]:
            logger.debug('Using tensorboard during training')
            log_path = os.path.join(current_checkpoint_path, 'tensorboard')
            callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, 
                                                                 histogram_freq=1))

        # training loop and save model at end of training
        serializer = ModelSerializer() 
        num_processors = CONFIG["training"]["NUM_PROCESSORS"]  
        epochs = CONFIG["training"]["EPOCHS"] 
        multiprocessing = num_processors > 1
        training = model.fit(train_data, epochs=epochs, validation_data=validation_data, 
                             callbacks=callbacks_list, workers=num_processors, 
                             use_multiprocessing=multiprocessing)

        serializer.save_pretrained_model(model, current_checkpoint_path)

        # save model parameters in json files         
        parameters = {'picture_shape' : CONFIG["model"]["IMG_SHAPE"],                           
                      'augmentation' : CONFIG["dataset"]["IMG_AUGMENT"],              
                      'batch_size' : CONFIG["training"]["BATCH_SIZE"],
                      'learning_rate' : CONFIG["training"]["LR_SCHEDULER"]["POST_WARMUP_LR"],
                      'epochs' : CONFIG["training"]["EPOCHS"],
                      'seed' : CONFIG["SEED"],
                      'tensorboard' : CONFIG["training"]["USE_TENSORBOARD"]}

        serializer.save_model_parameters(current_checkpoint_path, parameters)


    



