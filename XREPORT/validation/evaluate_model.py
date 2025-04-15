# [SETTING ENVIRONMENT VARIABLES]
from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.data.loader import InferenceDataLoader
from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.data.process.tokenizers import TokenWizard
from XREPORT.commons.utils.validation.reports import evaluation_report
from XREPORT.commons.utils.validation.checkpoints import ModelEvaluationSummary
from XREPORT.commons.utils.validation.text import EvaluateTextConsistency
from XREPORT.commons.constants import CONFIG, DATA_PATH
from XREPORT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':    

    # 1. [LOAD DATASET]
    #--------------------------------------------------------------------------  
    summarizer = ModelEvaluationSummary(CONFIG)    
    checkpoints_summary = summarizer.checkpoints_summary() 
    logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')  
    
    # 2. [LOAD MODEL]
    #--------------------------------------------------------------------------
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()             
    model, configuration, metadata, _, checkpoint_path = modelserializer.select_and_load_checkpoint()
    model.summary(expand_nested=True)
   
    # 3. [LOAD DATA]
    #--------------------------------------------------------------------------
    dataserializer = DataSerializer(configuration)
    train_data, val_data, metadata = dataserializer.load_train_and_validation_data()
    train_data = dataserializer.update_images_path(train_data)
    val_data = dataserializer.update_images_path(val_data)  
    vocabulary_size = metadata['vocabulary_size']
    logger.info(f'Train and validation data have been loaded')
    logger.info(f'Train samples: ({train_data.shape[0]} - Validation samples: {val_data.shape[0]})')
    logger.info(f'Vocabolary size: {vocabulary_size} tokens')    
       
    # 4. [EVALUATE ON TRAIN AND VALIDATION]
    #--------------------------------------------------------------------------  
    # use tf.data.Dataset to build the model dataloader with a larger batch size
    # the dataset is built on top of the training and validation data
    loader = InferenceDataLoader(CONFIG)        
    train_dataset = loader.build_inference_dataloader(train_data)
    validation_dataset = loader.build_inference_dataloader(val_data)

    # evaluate model performance over the training and validation dataset
    logger.info('Calculating model evaluation loss and metrics')    
    #evaluation_report(model, train_dataset, validation_dataset) 

    # 4. [CALCULATE SCORES]
    # One can select different either greedy_search or beam search to genarate
    # reports with a pretrained decoder 
    #--------------------------------------------------------------------------    
    scoring = EvaluateTextConsistency(model, configuration)
    scores = scoring.calculate_BLEU_score(train_data, val_data)    

    # 5. [TOKENIZERS]
    #--------------------------------------------------------------------------
    # get tokenizers and related configurations
    tokenization = TokenWizard(configuration)    
    tokenizer = tokenization.tokenizer

  
