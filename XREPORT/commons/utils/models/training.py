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
    def train_model(self, model, train_data, validation_data, current_checkpoint_path):

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
                      'learning_rate' : CONFIG["training"]["LEARNING_RATE"],
                      'epochs' : CONFIG["training"]["EPOCHS"],
                      'seed' : CONFIG["SEED"],
                      'tensorboard' : CONFIG["training"]["USE_TENSORBOARD"]}

        serializer.save_model_parameters(current_checkpoint_path, parameters)


    #--------------------------------------------------------------------------
    def calculate_loss(self, y_true, y_pred, mask):               
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss)/(tf.reduce_sum(mask) + keras.backend.epsilon())

        return loss


# [METRICS AND LOSSES]
#------------------------------------------------------------------------------
class TrainMetrics:


    def __init__(self, model : tf.keras.Model):


        self.model = model
        self.loss = model.loss 
        self.optimizer = model.optimizer

        # add loss and metrics tracker to average value for all batches
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.acc_tracker = keras.metrics.Mean(name='accuracy')  

    # calculate loss
    #--------------------------------------------------------------------------
    def calculate_loss(self, y_true, y_pred):              
                     
        loss = self.loss(y_true, y_pred)
        mask = tf.math.not_equal(y_true, 0)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss)/(tf.reduce_sum(mask) + keras.backend.epsilon())

        return loss
    
    # calculate accuracy
    #--------------------------------------------------------------------------
    def calculate_accuracy(self, y_true, y_pred): 
        y_true = tf.cast(y_true, dtype=tf.float32)        
        y_pred_argmax = tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.float32)
        accuracy = tf.equal(y_true, y_pred_argmax)
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.math.not_equal(y_true, 0)
        mask = tf.cast(mask, dtype=tf.float32)
        accuracy = tf.reduce_sum(accuracy) / (tf.reduce_sum(mask) + keras.backend.epsilon())

        return accuracy
    
    # define train step
    #--------------------------------------------------------------------------
    def train_step(self, batch_data):

        x_data, y_data = batch_data
        batch_img, batch_seq = x_data
        
        with tf.GradientTape() as tape:

            batch_seq_inp = batch_seq[:, :-1]
            batch_seq_true = batch_seq[:, 1:]             
            predictions = self.model(batch_img, batch_seq_inp)
            loss = self.calculate_loss(batch_seq_true, predictions)
            acc = self.calculate_accuracy(batch_seq_true, predictions)

        train_vars = self.model.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {'loss': self.loss_tracker.result(), 
                'accuracy': self.acc_tracker.result()}

    # define test step
    #--------------------------------------------------------------------------
    def test_step(self, batch_data):

        x_data, y_data = batch_data
        batch_img, batch_seq = x_data
        
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        predictions = self.model(batch_img, batch_seq_inp)
        loss = self.calculate_loss(batch_seq_true, predictions)
        acc = self.calculate_accuracy(batch_seq_true, predictions)
        
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {'loss': self.loss_tracker.result(), 
                'accuracy': self.acc_tracker.result()}
    
    #--------------------------------------------------------------------------
    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]   





