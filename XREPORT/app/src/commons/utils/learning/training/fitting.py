from keras import Model
from keras.utils import set_random_seed

from XREPORT.app.src.commons.utils.learning.callbacks import initialize_callbacks_handler
from XREPORT.app.src.commons.utils.data.serializer import ModelSerializer
from XREPORT.app.src.commons.logger import logger


           
# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:    
       
    def __init__(self, configuration, metadata=None):              
        set_random_seed(configuration.get('training_seed', 42))         
        self.configuration = configuration        
        self.metadata = metadata
        
    #--------------------------------------------------------------------------
    def train_model(self, model : Model, train_data, validation_data, metadata,
                    checkpoint_path, **kwargs): 
                
        total_epochs = self.configuration.get('epochs', 10)      
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, total_epochs=total_epochs, 
            progress_callback=kwargs.get('progress_callback', None), 
            worker=kwargs.get('worker', None))       
        
        # run model fit using keras API method.             
        session = model.fit(
            train_data, epochs=total_epochs, validation_data=validation_data, 
            callbacks=callbacks_list)

        serializer = ModelSerializer()  
        serializer.save_pretrained_model(model, checkpoint_path)       
        serializer.save_training_configuration(
            checkpoint_path, session, self.configuration, metadata)
        
    #--------------------------------------------------------------------------
    def resume_training(self, model : Model, train_data, validation_data, metadata,
                        checkpoint_path, session=None, **kwargs):
        
        from_epoch = 0 if not session else session['epochs']     
        total_epochs = from_epoch + self.configuration.get('additional_epochs', 10)           
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, session, total_epochs,
            progress_callback=kwargs.get('progress_callback', None), 
            worker=kwargs.get('worker', None))
        
        # run model fit using keras API method.             
        session = model.fit(
            train_data, epochs=total_epochs, validation_data=validation_data, 
            callbacks=callbacks_list, initial_epoch=from_epoch)

        serializer = ModelSerializer()  
        serializer.save_pretrained_model(model, checkpoint_path)       
        serializer.save_training_configuration(
            checkpoint_path, session, self.configuration, metadata)



    



