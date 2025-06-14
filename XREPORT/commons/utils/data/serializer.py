import os
import sys
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import keras

from XREPORT.commons.utils.data.database import XREPORTDatabase
from XREPORT.commons.utils.learning.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.utils.learning.scheduler import WarmUpLRScheduler
from XREPORT.commons.constants import CONFIG, DATA_PATH, METADATA_PATH, IMG_PATH, CHECKPOINT_PATH
from XREPORT.commons.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration):             
        self.img_shape = (224, 224)
        self.num_channels = 3               
        self.color_encoding = cv2.COLOR_BGR2RGB if self.num_channels == 3 else cv2.COLOR_BGR2GRAY
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}                     
        
        self.metadata_path = os.path.join(
            METADATA_PATH, 'preprocessing_metadata.json')         
        self.database = XREPORTDatabase(configuration)
        
        self.seed = configuration.get('general_seed', 42)         
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def load_source_dataset(self, sample_size=None):        
        dataset = self.database.load_source_data_table()
        sample_size = self.configuration.get('sample_size', 1.0)        
        dataset = dataset.sample(frac=sample_size, random_state=self.seed)     

        return dataset       

    # takes a reference dataset with images name and finds these images within the
    # image dataset directory, retriving their path accordingly
    #--------------------------------------------------------------------------
    def update_images_path(self, dataset):                
        images_path = {}
        for root, _, files in os.walk(IMG_PATH):                      
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:                                                       
                    path_pair = {file.split('.')[0] : os.path.join(IMG_PATH, file)}        
                    images_path.update(path_pair)         

        dataset['path'] = dataset['image'].map(images_path)
        dataset = dataset.dropna(subset=['path']).reset_index(drop=True)             

        return dataset
    
    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def get_images_path_from_directory(self, path, sample_size=1.0):          
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
    def load_train_and_validation_data(self): 
        # load preprocessed data from database and convert joint strings to list 
        train_data, val_data = self.database.load_train_and_validation_tables()

        # process text strings to obtain a list of separated token indices     
        train_data['tokens'] = train_data['tokens'].apply(
            lambda x : [int(f) for f in x.split()]) 
        val_data['tokens'] = val_data['tokens'].apply(
            lambda x : [int(f) for f in x.split()]) 
               
        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)        
        
        return train_data, val_data, metadata   

    #--------------------------------------------------------------------------
    def save_train_and_validation_data(self, train_data : pd.DataFrame, validation_data : pd.DataFrame,
                                       vocabulary_size=None):          
        self.database.save_train_and_validation_tables(train_data, validation_data)       
        metadata = {'seed' : self.configuration['SEED'], 
                    'dataset' : self.configuration['dataset'],
                    'date' : datetime.now().strftime("%Y-%m-%d"),
                    'vocabulary_size' : vocabulary_size}
                
        with open(self.metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4) 

    #--------------------------------------------------------------------------
    def save_generated_reports(self, data : pd.DataFrame):
        self.database.save_inference_data_table(data)  

    
    

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
        os.makedirs(os.path.join(checkpoint_path, 'data'), exist_ok=True)
        logger.debug(f'Created checkpoint folder at {checkpoint_path}')
        
        return checkpoint_path        

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model : keras.Model, path):
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

        logger.debug(f'Model configuration and session history saved for {os.path.basename(path)}')     

    #--------------------------------------------------------------------------
    def load_training_configuration(self, path): 
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
    def save_model_plot(self, model, path):       
        logger.debug('Generating model architecture graph')
        plot_path = os.path.join(path, 'model_layout.png')       
        keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                    show_layer_names=True, show_layer_activations=True, 
                    expand_nested=True, rankdir='TB', dpi=400)
        
    #--------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_name):    
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        custom_objects = {'MaskedSparseCategoricalCrossentropy': MaskedSparseCategoricalCrossentropy,
                          'MaskedAccuracy': MaskedAccuracy, 
                          'LRScheduler': WarmUpLRScheduler}     
                     
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name) 
        model_path = os.path.join(checkpoint_path, 'saved_model.keras') 
        model = keras.models.load_model(model_path, custom_objects=custom_objects)       
        configuration, session = self.load_training_configuration(checkpoint_path)        
            
        return model, configuration, session, checkpoint_path