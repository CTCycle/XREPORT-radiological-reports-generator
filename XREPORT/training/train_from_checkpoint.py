# [SETTING ENVIRONMENT VARIABLES]
from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.data.loader import TrainingDataLoader
from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.data.process.splitting import TrainValidationSplit
from XREPORT.commons.utils.learning.training import ModelTraining
from XREPORT.commons.utils.validation.reports import log_training_report

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PRETRAINED MODEL]
    #--------------------------------------------------------------------------    
    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models') 
    modelserializer = ModelSerializer()      
    model, configuration, metadata, _, checkpoint_path = modelserializer.select_and_load_checkpoint()    
    model.summary(expand_nested=True)  
    
    # setting device for training    
    trainer = ModelTraining(configuration)    
    trainer.set_device()  

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    # initialize training device, allows changing device prior to initializing the generators
    #--------------------------------------------------------------------------    
    # load saved tf.datasets from the proper folders in the checkpoint directory
    logger.info('Loading preprocessed data and building dataloaders')     
    dataserializer = DataSerializer(configuration) 
    train_data, val_data, metadata = dataserializer.load_train_and_validation_data()
    train_data = dataserializer.update_images_path(train_data)
    val_data = dataserializer.update_images_path(val_data)       
    
    # create the tf.datasets using the previously initialized generators 
    builder = TrainingDataLoader(configuration)   
    train_dataset, validation_dataset = builder.build_training_dataloader(
        train_data, val_data)    
    
    # 3. [TRAINING MODEL]  
    # Setting callbacks and training routine for the machine learning model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/     
    #--------------------------------------------------------------------------        
    log_training_report(train_data, val_data, configuration, metadata)                            

    # resume training from pretrained model    
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path,
                        from_checkpoint=True)


