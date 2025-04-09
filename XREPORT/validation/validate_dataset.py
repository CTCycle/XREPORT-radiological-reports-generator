# [SETTING ENVIRONMENT VARIABLES]
from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.data.serializer import DataSerializer
from XREPORT.commons.utils.data.process.splitting import TrainValidationSplit
from XREPORT.commons.utils.validation.images import ImageAnalysis
from XREPORT.commons.utils.validation.text import TextAnalysis
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATASET]
    #--------------------------------------------------------------------------  
    dataserializer = DataSerializer(CONFIG)
    dataset = dataserializer.load_source_dataset()   
    dataset = dataserializer.update_images_path(dataset)
    logger.info(f'Number of reports and related images: {dataset.shape[0]}')
     
    # 2. [COMPUTE IMAGE STATISTICS]
    #--------------------------------------------------------------------------
    analyzer = TextAnalysis()
    words = analyzer.count_words_in_documents(dataset)
    logger.info(f'Number of words dataset:        {len(words)}')
    logger.info(f'Number of unique words dataset: {len(set(words))}')     

    # 3. [COMPARE TRAIN AND TEST DATASETS]
    #--------------------------------------------------------------------------
    # load train and validation images as numpy arrays
    analyzer = ImageAnalysis(CONFIG)
    logger.info('Calculating image statistics and generating dataset report')
    logger.info('Focusing on mean pixel values, pixel standard deviation, image noise ratio')
    image_statistics = analyzer.calculate_image_statistics(dataset)

    logger.info('Generating the pixel intensity histogram')
    analyzer.calculate_pixel_intensity_distribution(dataset)   

    # 2. [SPLIT DATA]
    #--------------------------------------------------------------------------
    splitter = TrainValidationSplit(CONFIG, dataset)     
    train_data, validation_data = splitter.split_train_and_validation()
    logger.info('Splitting images pool into train and validation datasets')
    logger.info(f'Number of train samples: {len(train_data)}')
    logger.info(f'Number of validation samples: {len(validation_data)}')

