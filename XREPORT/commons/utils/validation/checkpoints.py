import os
import shutil
import pandas as pd

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
    def checkpoints_summary(self):       
        # look into checkpoint folder to get pretrained model names      
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for model_path in model_paths:            
            model = self.serializer.load_checkpoint(model_path)
            configuration, metadata, history = self.serializer.load_training_configurationn(model_path)
            model_name = os.path.basename(model_path)            

            # Extract model name and training type                       
            device_config = configuration["device"]
            precision = 16 if device_config.get("MIXED_PRECISION", 'NA') == True else 32           

            chkp_config = {'Checkpoint name': model_name,                                                  
                           'Sample size': configuration["dataset"].get("SAMPLE_SIZE", 'NA'),
                           'Validation size': configuration["dataset"].get("VALIDATION_SIZE", 'NA'),
                           'Seed': configuration.get("SEED", 'NA'),                           
                           'Precision (bits)': precision,                      
                           'Epochs': configuration["training"].get("EPOCHS", 'NA'),
                           'Additional Epochs': configuration["training"].get("ADDITIONAL_EPOCHS", 'NA'),
                           'Batch size': configuration["training"].get("BATCH_SIZE", 'NA'),           
                           'Split seed': configuration["dataset"].get("SPLIT_SEED", 'NA'),
                           'Image augmentation': configuration["dataset"].get("IMG_AUGMENTATION", 'NA'),
                           'Image height': 128,
                           'Image width': 128,
                           'Image channels': 3,                            
                           'JIT Compile': configuration["model"].get("JIT_COMPILE", 'NA'),
                           'JIT Backend': configuration["model"].get("JIT_BACKEND", 'NA'),
                           'Device': configuration["device"].get("DEVICE", 'NA'),
                           'Device ID': configuration["device"].get("DEVICE_ID", 'NA'),                           
                           'Number of Processors': configuration["device"].get("NUM_PROCESSORS", 'NA'),
                           'Use TensorBoard': configuration["training"].get("USE_TENSORBOARD", 'NA'),                            
                           'LR Scheduler - Post Warmup LR': configuration["training"]["LR_SCHEDULER"].get("POST_WARMUP_LR", 'NA'),
                           'LR Scheduler - Warmup Steps': configuration["training"]["LR_SCHEDULER"].get("WARMUP_STEPS", 'NA'),
                           'Temperature': configuration["training"].get("TEMPERATURE", 'NA'),                            
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
    
    
