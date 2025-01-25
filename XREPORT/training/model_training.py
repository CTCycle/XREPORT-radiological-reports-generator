# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.process.tokenizers import TokenWizard
from XREPORT.commons.utils.dataloader.tensordata import TensorDatasetBuilder
from XREPORT.commons.utils.process.splitting import TrainValidationSplit
from XREPORT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
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
    # load data from csv, add paths to images 
    dataserializer = DataSerializer(CONFIG)
    processed_data, metadata = dataserializer.load_preprocessed_data() 
    processed_data = dataserializer.get_images_path_from_dataset(processed_data)
    vocabulary_size = metadata['vocabulary_size']

    # 2. [SPLIT DATA]
    #--------------------------------------------------------------------------
    # split data into train set and validation set
    logger.info('Preparing dataset of images and captions based on splitting size')  
    splitter = TrainValidationSplit(CONFIG, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation()    

    # create subfolder for preprocessing data
    modelserializer = ModelSerializer()
    checkpoint_path = modelserializer.create_checkpoint_folder()
    dataserializer.copy_data_to_checkpoint(checkpoint_path)       

    # 3. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    logger.info('Building XREPORT model and data loaders')     
    trainer = ModelTraining(CONFIG) 
    trainer.set_device()

    # get tokenizers from preprocessing configurations
    tokenization = TokenWizard(metadata)   
    tokenizer = tokenization.tokenizer
       
    # create the tf.datasets using the previously initialized generators 
    builder = TensorDatasetBuilder(CONFIG)   
    train_dataset, validation_dataset = builder.build_model_dataloader(train_data, validation_data)      

    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/
    log_training_report(train_data, validation_data, CONFIG, vocabulary_size=vocabulary_size)

    # initialize and compile the captioning model    
    captioner = XREPORTModel(vocabulary_size, CONFIG)
    model = captioner.get_model(model_summary=True) 

    # generate graphviz plot fo the model layout       
    modelserializer.save_model_plot(model, checkpoint_path)              

    # perform training and save model at the end
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path)
        
    



