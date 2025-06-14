import os
import shutil
import pandas as pd

from XREPORT.commons.utils.learning.callbacks import InterruptTraining
from XREPORT.commons.utils.data.database import XREPORTDatabase
from XREPORT.commons.utils.data.serializer import ModelSerializer
from XREPORT.commons.constants import CHECKPOINT_PATH
from XREPORT.commons.logger import logger


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, configuration, remove_invalid=False):
        self.remove_invalid = remove_invalid
        self.serializer = ModelSerializer()        
        self.database = XREPORTDatabase(configuration)        
        self.configuration = configuration

    #---------------------------------------------------------------------------
    def scan_checkpoint_folder(self):
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():                
                pretrained_model_path = os.path.join(entry.path, 'saved_model.keras')                
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)
                elif not os.path.isfile(pretrained_model_path) and self.remove_invalid:                    
                    shutil.rmtree(entry.path)

        return model_paths  

    #---------------------------------------------------------------------------
    def get_checkpoints_summary(self):            
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for model_path in model_paths:            
            model = self.serializer.load_checkpoint(model_path)
            configuration, history = self.serializer.load_training_configuration(model_path)
            model_name = os.path.basename(model_path)                   
            precision = 16 if configuration.get("use_mixed_precision", 'NA') else 32 
            chkp_config = {'Sample size': configuration.get("train_sample_size", 'NA'),
                           'Validation size': configuration.get("validation_size", 'NA'),
                           'Seed': configuration.get("train_seed", 'NA'),                           
                           'Precision (bits)': precision,                      
                           'Epochs': configuration.get("epochs", 'NA'),
                           'Additional Epochs': configuration.get("additional_epochs", 'NA'),
                           'Batch size': configuration.get("batch_size", 'NA'),           
                           'Split seed': configuration.get("split_seed", 'NA'),
                           'Image augmentation': configuration.get("img_augmentation", 'NA'),
                           'Image height': 224,
                           'Image width': 224,
                           'Image channels': 3,                          
                           'JIT Compile': configuration.get("jit_compile", 'NA'),                           
                           'Device': configuration.get("device", 'NA'),                                                      
                           'Number workers': configuration.get("num_workers", 'NA'),
                           'LR Scheduler': configuration.get("use_scheduler", 'NA'),                            
                           'LR Scheduler - Post Warmup LR': configuration.get("post_warmup_LR", 'NA'),
                           'LR Scheduler - Warmup Steps': configuration.get("warmup_steps", 'NA'),
                           'Temperature': configuration.get("train_temperature", 'NA'),                            
                           'Tokenizer': configuration["dataset"].get("TOKENIZER", 'NA'),                            
                           'Max report size': configuration["dataset"].get("MAX_REPORT_SIZE", 'NA'),
                           'Number of heads': configuration["model"].get("ATTENTION_HEADS", 'NA'),
                           'Number of encoders': configuration["model"].get("NUM_ENCODERS", 'NA'),
                           'Number of decoders': configuration["model"].get("NUM_DECODERS", 'NA'),
                           'Embedding dimensions': configuration["model"].get("EMBEDDING_DIMS", 'NA'),
                           'Frozen image encoder': configuration["model"].get("FREEZE_IMG_ENCODER", 'NA')}

            model_parameters.append(chkp_config)

        dataframe = pd.DataFrame(model_parameters)
        self.database.save_checkpoints_summary_table(dataframe)        
            
        return dataframe
    
    #--------------------------------------------------------------------------
    def get_evaluation_report(self, model, validation_dataset, progress_callback=None, worker=None):
        callbacks_list = [InterruptTraining(worker)]
        validation = model.evaluate(validation_dataset, verbose=1, callbacks=callbacks_list)    
        logger.info(
            f'RMSE loss {validation[0]:.3f} - Cosine similarity {validation[1]:.3f}')     
    
    
