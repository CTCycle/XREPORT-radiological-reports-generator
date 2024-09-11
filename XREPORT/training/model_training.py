# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.preprocessing.tokenizers import BERTokenizer
from XREPORT.commons.utils.dataloader.generators import training_data_pipeline
from XREPORT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.learning.training import ModelTraining
from XREPORT.commons.utils.learning.models import XREPORTModel
from XREPORT.commons.constants import CONFIG, ML_DATA_PATH
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PREPROCESSED DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images 
    dataserializer = DataSerializer()
    train_data, validation_data, metadata = dataserializer.load_preprocessed_data(ML_DATA_PATH)    

    # create subfolder for preprocessing data, move preprocessed data to the 
    # checkpoint subfolder checkpoint/data   
    modelserializer = ModelSerializer()
    model_folder_path = modelserializer.create_checkpoint_folder() 
    modelserializer.store_data_in_checkpoint_folder(model_folder_path)   

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    logger.info('Building XREPORT model and data loaders')     
    trainer = ModelTraining(CONFIG) 
    trainer.set_device()

    # get tokenizers and its info
    tokenization = BERTokenizer()    
    tokenizer = tokenization.tokenizer
       
    # create the tf.datasets using the previously initialized generators    
    train_dataset, validation_dataset = training_data_pipeline(train_data, validation_data)
    vocab_size = len(tokenizer.vocab) + 1   

    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/
    logger.info('XREPORT training report')
    logger.info('--------------------------------------------------------------') 
    logger.info(f'Number of train samples:      {len(train_data)}')
    logger.info(f'Number of validation samples: {len(validation_data)}')      
    logger.info(f'Batch size:                   {CONFIG["training"]["BATCH_SIZE"]}')
    logger.info(f'Epochs:                       {CONFIG["training"]["EPOCHS"]}')
    logger.info(f'Vocabulary size:              {vocab_size}')
    logger.info(f'Max caption length:           {CONFIG["dataset"]["MAX_REPORT_SIZE"]}')
    logger.info('--------------------------------------------------------------')    

    # initialize and compile the captioning model    
    captioner = XREPORTModel(vocab_size)
    model = captioner.get_model(summary=True) 

    # generate graphviz plot fo the model layout       
    modelserializer.save_model_plot(model, model_folder_path)              

    # perform training and save model at the end
    trainer.train_model(model, train_dataset, validation_dataset, model_folder_path)
        
    



