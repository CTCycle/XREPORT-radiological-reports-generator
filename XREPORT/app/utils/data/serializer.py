import os
import sys
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from keras.utils import plot_model
from keras.models import load_model

from XREPORT.app.utils.data.database import XREPORTDatabase
from XREPORT.app.utils.learning.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.app.utils.learning.training.scheduler import WarmUpLRScheduler
from XREPORT.app.constants import METADATA_PATH, IMG_PATH, CHECKPOINT_PATH
from XREPORT.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration : dict):             
        self.img_shape = (224, 224)
        self.num_channels = 3               
        self.color_encoding = cv2.COLOR_BGR2RGB if self.num_channels == 3 else cv2.COLOR_BGR2GRAY
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}                     
        # define metadata path and extract configuration parameters
        self.metadata_path = os.path.join(METADATA_PATH, 'preprocessing_metadata.json')        
        self.seed = configuration.get('seed', 42) 
        self.max_report_size = configuration.get('max_report_size', 200) 
        self.tokenizer_ID = configuration.get('tokenizer', None)
        # create database instance
        self.database = XREPORTDatabase()                 
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def load_source_dataset(self, sample_size=1.0):        
        dataset = self.database.load_source_dataset()
        if sample_size < 1.0:
            dataset = dataset.sample(frac=sample_size, random_state=self.seed)     

        return dataset       

    # takes a reference dataset with images name and finds these images within the
    # image dataset directory, retriving their path accordingly
    #--------------------------------------------------------------------------
    def update_img_path(self, dataset : pd.DataFrame):                
        images_path = {}
        for root, _, files in os.walk(IMG_PATH):                      
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:                                                       
                    path_pair = {file.split('.')[0] : os.path.join(IMG_PATH, file)}        
                    images_path.update(path_pair)         

        dataset['path'] = dataset['image'].map(images_path)
        clean_dataset = dataset.dropna(subset=['path']).reset_index(drop=True) 
        logger.info(f'Updated dataset with images paths: {len(clean_dataset)} records found')  
        logger.info(f'{len(dataset) - len(clean_dataset)} records were dropped due to missing images')          

        return clean_dataset
    
    #--------------------------------------------------------------------------
    def get_img_path_from_directory(self, path, sample_size=1.0):          
        if not os.listdir(path):
            logger.error(f'No images found in {path}, please add them and try again.')
            sys.exit()
        else:            
            logger.debug(f'Valid extensions are: {self.valid_extensions}')
            images_path = []
            for root, _, files in os.walk(path):
                if sample_size < 1.0:
                    files = files[:int(sample_size * len(files))]           
                for file in files:                
                    if os.path.splitext(file)[1].lower() in self.valid_extensions:
                        images_path.append(os.path.join(root, file))                

            return images_path
        
    #--------------------------------------------------------------------------
    def validate_metadata(self, metadata : dict, target_metadata : dict):        
        keys_to_compare = [k for k in metadata if k != "date"]
        meta_current = {k: metadata.get(k) for k in keys_to_compare}
        meta_target = {k: target_metadata.get(k) for k in keys_to_compare}        
        differences = {k: (meta_current[k], meta_target[k]) 
                       for k in keys_to_compare if meta_current[k] != meta_target[k]} 
        
        return False if differences else True
        
    #--------------------------------------------------------------------------
    def process_tokens(self, col):        
        if isinstance(col, list):
            return ' '.join(map(str, col))
        if isinstance(col, str):
            return [int(f) for f in col.split() if f.strip()]
        return []

    #--------------------------------------------------------------------------
    def load_train_and_validation_data(self, only_metadata=False):
        # load metadata from file
        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)     

        if not only_metadata: 
            # load preprocessed data from database and convert joint strings to list 
            train_data, val_data = self.database.load_train_and_validation()        
            # process text strings to obtain a list of separated token indices     
            train_data['tokens'] = train_data['tokens'].apply(self.process_tokens)             
            val_data['tokens'] = val_data['tokens'].apply(self.process_tokens)      
        
            return train_data, val_data, metadata  

        return metadata 

    #--------------------------------------------------------------------------
    def save_train_and_validation_data(self, train_data : pd.DataFrame, validation_data : pd.DataFrame,
                                       vocabulary_size=None): 
        # process list of tokens to get them in string format   
        train_data['tokens'] = train_data['tokens'].apply(self.process_tokens)             
        validation_data['tokens'] = validation_data['tokens'].apply(self.process_tokens)
        # save train and validation data to database             
        self.database.save_train_and_validation(train_data, validation_data)
        # save preprocessing metadata
        metadata = {'seed' : self.seed,                     
                    'date' : datetime.now().strftime("%Y-%m-%d"),
                    'sample_size' : self.configuration.get('sample_size', 1.0),
                    'validation_size' : self.configuration.get('validation_size', 0.2),
                    'split_seed' : self.configuration.get('split_seed', 42),
                    'vocabulary_size' : vocabulary_size,
                    'max_report_size' : self.max_report_size,
                    'tokenizer' : self.tokenizer_ID}
                
        with open(self.metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4) 

    #--------------------------------------------------------------------------
    def save_generated_reports(self, reports : list[dict]):        
        reports_dataframe = pd.DataFrame(reports)
        self.database.save_generated_reports(reports_dataframe)

    #--------------------------------------------------------------------------
    def save_text_statistics(self, data : pd.DataFrame):            
        self.database.save_text_statistics(data)  
    
    #--------------------------------------------------------------------------
    def save_image_statistics(self, data : pd.DataFrame):            
        self.database.save_image_statistics(data)       

    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame):            
        self.database.save_checkpoints_summary(data)
    

# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'XREPORT'

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):              
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_path = os.path.join(
            CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
        os.makedirs(checkpoint_path, exist_ok=True)   
        logger.debug(f'Created checkpoint folder at {checkpoint_path}')
        
        return checkpoint_path        

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model, path : str):
        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model {os.path.basename(path)} has been saved')

    #--------------------------------------------------------------------------
    def save_training_configuration(self, path, history : dict, configuration : dict, metadata : dict):         
        os.makedirs(os.path.join(path, 'configuration'), exist_ok=True)         
        config_path = os.path.join(path, 'configuration', 'configuration.json')
        metadata_path = os.path.join(path, 'configuration', 'metadata.json') 
        history_path = os.path.join(path, 'configuration', 'session_history.json')
        # Save training and model configuration
        with open(config_path, 'w') as f:
            json.dump(configuration, f)
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f) 
        # Save session history
        with open(history_path, 'w') as f:
            json.dump(history, f)

        logger.debug(f'Model configuration, session history and metadata saved for {os.path.basename(path)}')     

    #--------------------------------------------------------------------------
    def load_training_configuration(self, path : str): 
        config_path = os.path.join(path, 'configuration', 'configuration.json')        
        with open(config_path, 'r') as f:
            configuration = json.load(f) 

        config_path = os.path.join(path, 'configuration', 'metadata.json')        
        with open(config_path, 'r') as f:
            metadata = json.load(f)       

        history_path = os.path.join(path, 'configuration', 'session_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configuration, metadata, history  

    #-------------------------------------------------------------------------- 
    def scan_checkpoints_folder(self):
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)
        
        return model_folders      

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path : str):       
        logger.debug('Generating model architecture graph')
        plot_path = os.path.join(path, 'model_layout.png')       
        plot_model(model, to_file=plot_path, show_shapes=True, 
                    show_layer_names=True, show_layer_activations=True, 
                    expand_nested=True, rankdir='TB', dpi=400)
        
    #--------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_name : str): 
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        custom_objects = {'MaskedSparseCategoricalCrossentropy': MaskedSparseCategoricalCrossentropy,
                          'MaskedAccuracy': MaskedAccuracy, 
                          'LRScheduler': WarmUpLRScheduler}     
                     
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name) 
        model_path = os.path.join(checkpoint_path, 'saved_model.keras')         
        model = load_model(model_path, custom_objects=custom_objects)       
        configuration, metadata, session = self.load_training_configuration(checkpoint_path)        
            
        return model, configuration, metadata, session, checkpoint_path