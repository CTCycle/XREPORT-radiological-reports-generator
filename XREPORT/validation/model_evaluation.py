import random 

# [SETTING ENVIRONMENT VARIABLES]
from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.validation.reports import DataAnalysisPDF
from FEXT.commons.utils.dataloader.tensordata import TensorDatasetBuilder
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.process.splitting import TrainValidationSplit
from FEXT.commons.utils.inference.encoding import ImageEncoding
from FEXT.commons.utils.validation.images import ImageReconstruction
from FEXT.commons.utils.validation.checkpoints import ModelEvaluationSummary
from FEXT.commons.utils.validation.reports import evaluation_report
from FEXT.commons.constants import CONFIG, IMG_DATA_PATH, VALIDATION_PATH
from FEXT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    evaluation_batch_size = 64
    num_images_to_evaluate = 6

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

    # isolate the encoder from the autoencoder model   
    encoder = ImageEncoding(model, configuration)
    encoder_model = encoder.encoder_model   

    # 3. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------
    dataserializer = DataSerializer(configuration)
    images_path = dataserializer.get_images_path(IMG_DATA_PATH)

    splitter = TrainValidationSplit(images_path, configuration)     
    train_data, validation_data = splitter.split_train_and_validation()    

    builder = TensorDatasetBuilder(configuration)      
    train_dataset, validation_dataset = builder.build_model_dataloader(
        train_data, validation_data, evaluation_batch_size)    

    # 4. [EVALUATE ON TRAIN AND VALIDATION]
    #--------------------------------------------------------------------------   
    evaluation_report(model, train_dataset, validation_dataset) 

    # 5. [COMPARE RECONTRUCTED IMAGES]
    #--------------------------------------------------------------------------
    validator = ImageReconstruction(model, checkpoint_path)
    train_images_batch = [
        dataserializer.load_image(path) for path in 
        random.sample(train_data, num_images_to_evaluate)]
    validation_images_batch = [
        dataserializer.load_image(path) for path in 
        random.sample(validation_data, num_images_to_evaluate)]
    
    logger.info('Generating comparison of reconstructed images versus input from train dataset')
    validator.visualize_reconstructed_images(train_images_batch, data_name='train')
    logger.info('Generating comparison of reconstructed images versus input from validation dataset')
    validator.visualize_reconstructed_images(validation_images_batch, data_name='validation')

    # 2. [INITIALIZE PDF REPORT]
    #--------------------------------------------------------------------------
    report = DataAnalysisPDF()
