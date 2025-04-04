import keras

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


###############################################################################
def evaluation_report(model : keras.Model, train_dataset, validation_dataset):    
    training = model.evaluate(train_dataset, verbose=1)
    validation = model.evaluate(validation_dataset, verbose=1)
    logger.info(
        f'Training loss {training[0]:.3f} - Training metric {training[1]:.3f}')    
    logger.info(
        f'Validation loss {validation[0]:.3f} - Validation metric {validation[1]:.3f}')  
    

###############################################################################
def log_training_report(train_data, validation_data, config : dict, metadata : dict):
    vocabulary_size = metadata['vocabulary_size']
    logger.info('--------------------------------------------------------------')
    logger.info('XREPORT training report')
    logger.info('--------------------------------------------------------------')    
    logger.info(f'Number of train samples:       {len(train_data)}')
    logger.info(f'Number of validation samples:  {len(validation_data)}')
    logger.info(f'Vocabulary size:               {vocabulary_size}')    
    for key, value in config.items():
        if isinstance(value, dict) and 'validation' not in key:
            for sub_key, sub_value in value.items():
                if sub_key == 'ADDITIONAL_EPOCHS':
                    sub_value = CONFIG['training']['ADDITIONAL_EPOCHS']                
                if isinstance(sub_value, dict):
                    for inner_key, inner_value in sub_value.items():
                        logger.info(f'{key}.{sub_key}.{inner_key}: {inner_value}')
                else:
                    logger.info(f'{key}.{sub_key}: {sub_value}')
        elif 'validation' not in key:
            logger.info(f'{key}: {value}')

    logger.info('--------------------------------------------------------------\n')



