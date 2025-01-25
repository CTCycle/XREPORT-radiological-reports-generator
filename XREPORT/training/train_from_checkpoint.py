# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.dataloader.tensordata import TensorDatasetBuilder
from XREPORT.commons.utils.process.splitting import TrainValidationSplit
from XREPORT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
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
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()    
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
    processed_data, metadata = dataserializer.load_data_from_checkpoint(checkpoint_path)
    processed_data = dataserializer.get_images_path_from_dataset(processed_data)
    vocabulary_size = metadata['vocabulary_size']

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators
    splitter = TrainValidationSplit(configuration, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation()        
    
    # create the tf.datasets using the previously initialized generators 
    builder = TensorDatasetBuilder(configuration)   
    train_dataset, validation_dataset = builder.build_model_dataloader(train_data, validation_data)    
    
    # 3. [TRAINING MODEL]  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/     
    #--------------------------------------------------------------------------    
    log_training_report(train_data, validation_data, configuration, 
                        vocabulary_size=vocabulary_size, from_checkpoint=True)    

    # resume training from pretrained model    
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path,
                        from_checkpoint=True)


