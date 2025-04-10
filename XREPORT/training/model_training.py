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
from XREPORT.commons.utils.learning.models import XREPORTModel
from XREPORT.commons.utils.validation.reports import log_training_report
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PREPROCESSED DATA]
    #--------------------------------------------------------------------------     
    # load data from csv
    dataserializer = DataSerializer(CONFIG)
    processed_data, metadata = dataserializer.load_processed_data() 
    # fetch images path from the preprocessed data
    processed_data = dataserializer.get_training_images_path(processed_data)
    vocabulary_size = metadata['vocabulary_size']

    # 2. [SPLIT DATA]
    #--------------------------------------------------------------------------
    # split data into train set and validation set
    logger.info('Preparing dataset of images and captions based on splitting size')  
    splitter = TrainValidationSplit(CONFIG, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation()              

    # 3. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    logger.info('Building model data loaders with prefetching and parallel processing') 
    builder = TrainingDataLoader(CONFIG)   
    train_dataset, validation_dataset = builder.build_training_dataloader(
        train_data, validation_data) 
    
    modelserializer = ModelSerializer()
    checkpoint_path = modelserializer.create_checkpoint_folder() 

    # 4. [SET DEVICE]
    #--------------------------------------------------------------------------
    logger.info('Setting device for training operations based on user configurations')       
    trainer = ModelTraining(CONFIG, metadata) 
    trainer.set_device()              

    # 5. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the machine learning model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/
    log_training_report(train_data, validation_data, CONFIG, metadata)

    # initialize and compile the captioning model    
    logger.info('Building XREPORT Transformer model based on user configurations')
    captioner = XREPORTModel(vocabulary_size, CONFIG)
    model = captioner.get_model(model_summary=True) 

    # generate graphviz plot for the model layout       
    modelserializer.save_model_plot(model, checkpoint_path)              

    # perform training and save model at the end
    logger.info('Starting XREPORT Transformer model training') 
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path)
        
    



