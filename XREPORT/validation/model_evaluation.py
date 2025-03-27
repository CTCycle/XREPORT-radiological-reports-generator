# [SETTING ENVIRONMENT VARIABLES]
from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.data.loader import TrainingDataLoader
from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.data.process.tokenizers import TokenWizard
from XREPORT.commons.utils.data.process.splitting import TrainValidationSplit
from XREPORT.commons.utils.validation.reports import evaluation_report
from XREPORT.commons.utils.validation.checkpoints import ModelEvaluationSummary
from XREPORT.commons.constants import CONFIG, DATA_PATH
from XREPORT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    evaluation_batch_size = 20   

    # 1. [LOAD DATASET]
    #--------------------------------------------------------------------------  
    summarizer = ModelEvaluationSummary()    
    checkpoints_summary = summarizer.checkpoints_summary() 
    logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')  
    
    # 2. [LOAD MODEL]
    #--------------------------------------------------------------------------
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()
    model.summary(expand_nested=True)
   
    # 3. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------
    dataserializer = DataSerializer(configuration)
    processed_data, metadata = dataserializer.load_preprocessed_data()
    processed_data = dataserializer.get_images_path_from_dataset(processed_data)
    vocabulary_size = metadata['vocabulary_size']

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators
    splitter = TrainValidationSplit(configuration, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation() 

    # 4. [BUILD DATA LOADERS]
    #--------------------------------------------------------------------------
    # get tokenizers and related configurations
    tokenization = TokenWizard(configuration)    
    tokenizer = tokenization.tokenizer
    
    # 5. [EVALUATE ON TRAIN AND VALIDATION]
    #--------------------------------------------------------------------------  
    # use tf.data.Dataset to build the model dataloader with a larger batch size
    # the dataset is built on top of the training and validation data
    builder = TrainingDataLoader(CONFIG, evaluate=True)        
    train_dataset, validation_dataset = builder.build_model_dataloader(
        train_data, validation_data)

    # evaluate model performance over the training and validation dataset    
    evaluation_report(model, train_dataset, validation_dataset) 

  
