import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

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
    def train_model(self, model : tf.keras.Model, train_data, 
                    validation_data, current_checkpoint_path):

        # initialize the real time history callback    
        RTH_callback = RealTimeHistory(current_checkpoint_path, validation=True)
        callbacks_list = [RTH_callback]

        # initialize tensorboard if requested    
        if CONFIG["training"]["USE_TENSORBOARD"]:
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


 



# [LOSS FUNCTION]
#------------------------------------------------------------------------------
class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    
    def __init__(self, name='masked_SCC'):
        super().__init__(name=name)
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, 
                                                               reduction=keras.losses.Reduction.NONE)
        
    #--------------------------------------------------------------------------
    @tf.function
    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.math.not_equal(y_true, 0)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss)/(tf.reduce_sum(mask) + keras.backend.epsilon())

        return loss
    
    
# [METRICS]
#------------------------------------------------------------------------------
class MaskedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    #--------------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred_argmax = tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.float32)
        accuracy = tf.equal(y_true, y_pred_argmax)        
        # Create a mask to ignore padding (assuming padding value is 0)
        mask = tf.math.not_equal(y_true, 0)        
        # Apply the mask to the accuracy
        accuracy = tf.math.logical_and(mask, accuracy)        
        # Cast the boolean values to float32
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            accuracy = tf.multiply(accuracy, sample_weight)
            mask = tf.multiply(mask, sample_weight)
        
        # Update the state variables
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.reduce_sum(mask))
    
    #--------------------------------------------------------------------------
    def result(self):
        return self.total / (self.count + K.epsilon())
    
    #--------------------------------------------------------------------------
    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)







