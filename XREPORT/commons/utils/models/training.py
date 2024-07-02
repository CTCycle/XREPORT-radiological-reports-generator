import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from XREPORT.commons.utils.models.callbacks import RealTimeHistory
from XREPORT.commons.utils.dataloader.serializer import ModelSerializer
from XREPORT.commons.constants import CONFIG


           
# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
#------------------------------------------------------------------------------
class ModelTraining:    
       
    def __init__(self):                            
        np.random.seed(CONFIG["SEED"])
        tf.random.set_seed(CONFIG["SEED"])         
        self.available_devices = tf.config.list_physical_devices()               
        print('The current devices are available:\n')        
        for dev in self.available_devices:            
            print(dev)

    # set device
    #--------------------------------------------------------------------------
    def set_device(self):
       
        if CONFIG["training"]["ML_DEVICE"] == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                print('\nNo GPU found. Falling back to CPU\n')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if CONFIG["training"]["MIXED_PRECISION"]:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
                os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'                 
                print('\nGPU is set as active device\n')
                   
        elif CONFIG["training"]["ML_DEVICE"] == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('\nCPU is set as active device\n')    

    #--------------------------------------------------------------------------
    def training_session(self, model, train_data, validation_data, current_checkpoint_path):


        serializer = ModelSerializer()

        # initialize the real time history callback    -
        RTH_callback = RealTimeHistory(current_checkpoint_path, validation=True)
        callbacks_list = [RTH_callback]

        # initialize tensorboard if requested    
        if USE_TENSORBOARD:
            log_path = os.path.join(current_checkpoint_path, 'tensorboard')
            callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, 
                                                                 histogram_freq=1))

        # training loop and save model at end of training    
        multiprocessing = NUM_PROCESSORS > 1
        training = model.fit(train_data, epochs=EPOCHS, validation_data=validation_data, 
                            callbacks=callbacks_list, workers=NUM_PROCESSORS, 
                            use_multiprocessing=multiprocessing)

        model_files_path = os.path.join(current_checkpoint_path, 'model')
        serializer.save_XREPORT_model(model, model_files_path)
        print(f'\nTraining session is over. Model has been saved in folder {current_checkpoint_path}')

        # save model parameters in json files            
        parameters = {'picture_shape' : IMG_SHAPE,                           
                      'augmentation' : IMG_AUGMENT,              
                      'batch_size' : BATCH_SIZE,
                      'learning_rate' : LEARNING_RATE,
                      'epochs' : EPOCHS,
                      'seed' : SEED,
                      'tensorboard' : USE_TENSORBOARD}

        serializer.save_model_parameters(current_checkpoint_path, parameters)
    

