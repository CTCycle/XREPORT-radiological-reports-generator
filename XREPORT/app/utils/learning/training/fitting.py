from keras import Model
from keras.utils import set_random_seed

from XREPORT.app.utils.learning.callbacks import initialize_callbacks_handler


           
# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:    
       
    def __init__(self, configuration : dict, metadata=None):              
        set_random_seed(configuration.get('training_seed', 42))         
        self.configuration = configuration        
        self.metadata = metadata
        
    #--------------------------------------------------------------------------
    def train_model(self, model : Model, train_data, validation_data, checkpoint_path, **kwargs):                 
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
        
        history = {'history' : session.history,
                   'epochs': session.epoch[-1] + 1} 

        return model, history       
        
    #--------------------------------------------------------------------------
    def resume_training(self, model : Model, train_data, validation_data, 
        checkpoint_path, session=None, additional_epochs=10, **kwargs):
        from_epoch = 0 if not session else session['epochs']     
        total_epochs = from_epoch + additional_epochs            
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, session, total_epochs,
            progress_callback=kwargs.get('progress_callback', None), 
            worker=kwargs.get('worker', None))
        
        # run model fit using keras API method.             
        new_session = model.fit(
            train_data, epochs=total_epochs, validation_data=validation_data, 
            callbacks=callbacks_list, initial_epoch=from_epoch)     

        # update history with new scores and final epoch value
        session_keys = session['history'].keys() 
        new_history = {k: session['history'][k] + new_session.history[k] for k in session_keys}
        history = {'history' : new_history,
                   'epochs': new_session.epoch[-1] + 1}  

        return model, history



    



