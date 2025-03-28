import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
import keras

from XREPORT.commons.utils.data.database import XREPORTDatabase
from XREPORT.commons.utils.learning.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.utils.learning.scheduler import LRScheduler
from XREPORT.commons.constants import CONFIG, DATA_PATH, METADATA_PATH, IMG_PATH, CHECKPOINT_PATH
from XREPORT.commons.logger import logger


###############################################################################
def checkpoint_selection_menu(models_list):

    index_list = [idx + 1 for idx, item in enumerate(models_list)]     
    print('Currently available pretrained models:')             
    for i, directory in enumerate(models_list):
        print(f'{i + 1} - {directory}')                         
    while True:
        try:
            selection_index = int(input('\nSelect the pretrained model: '))
            print()
        except ValueError:
            logger.error('Invalid choice for the pretrained model, asking again')
            continue
        if selection_index in index_list:
            break
        else:
            logger.warning('Model does not exist, please select a valid index')

    return selection_index


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
        
        self.metadata_path = os.path.join(METADATA_PATH, 'XREPORT_metadata.json')         
        self.database = XREPORTDatabase(configuration)
        
        self.seed = configuration['SEED']   
        self.parameters = configuration["dataset"]        
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def load_dataset(self, sample_size=None):        
        dataset = self.database.load_source_data()
        sample_size = self.parameters["SAMPLE_SIZE"] if sample_size is None else sample_size        
        dataset = dataset.sample(frac=sample_size, random_state=self.seed)     

        return dataset

    # takes a reference dataset with images name and finds these images within the
    # image dataset directory, retriving their path accordingly
    #--------------------------------------------------------------------------
    def get_training_images_path(self, dataset : pd.DataFrame):                
        images_path = {}
        for root, _, files in os.walk(IMG_PATH):                      
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:                                                       
                    path_pair = {file.split('.')[0] : os.path.join(IMG_PATH, file)}        
                    images_path.update(path_pair)         

        dataset['path'] = dataset['id'].map(images_path)
        dataset = dataset.dropna(subset=['path']).reset_index(drop=True)             

        return dataset
    
    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def get_images_path_from_directory(self, path):           
        if not os.listdir(path):
            logger.error(f'No images found in {path}, please add them and try again.')
            sys.exit()
        else:            
            logger.debug(f'Valid extensions are: {self.valid_extensions}') 
            images_path = []
            for root, _, files in os.walk(path):                 
                for file in files:
                    if os.path.splitext(file)[1].lower() in self.valid_extensions:
                        images_path.append(os.path.join(root, file))                

            return images_path    

    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, processed_data : pd.DataFrame, vocabulary_size=None):               
        self.database.save_preprocessed_data(processed_data)      
        metadata = {'seed' : self.configuration['SEED'], 
                    'dataset' : self.configuration['dataset'],
                    'date' : datetime.now().strftime("%Y-%m-%d"),
                    'vocabulary_size' : vocabulary_size}
                
        with open(self.metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)         

    #--------------------------------------------------------------------------
    def load_preprocessed_data(self): 
        # load preprocessed data from database table 
        processed_data = self.database.load_preprocessed_data()
        # process text strings to obtain a list of separated token indices     
        processed_data['tokens'] = processed_data['tokens'].apply(
            lambda x : [int(f) for f in x.split()]) 
               
        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)        
        
        return processed_data, metadata        
    
    

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
        logger.info(f'Training session is over. Model has been saved in folder {path}')

    #--------------------------------------------------------------------------
    def save_session_configuration(self, path, history : dict, configurations : dict):         
        os.makedirs(os.path.join(path, 'configurations'), exist_ok=True)         
        config_path = os.path.join(path, 'configurations', 'configurations.json')
        history_path = os.path.join(path, 'configurations', 'session_history.json')        

        # Save training and model configurations
        with open(config_path, 'w') as f:
            json.dump(configurations, f)       

        # Save session history
        with open(history_path, 'w') as f:
            json.dump(history, f)

        logger.debug(f'Model configuration and session history saved for {os.path.basename(path)}')     

    #--------------------------------------------------------------------------
    def load_session_configuration(self, path): 
        config_path = os.path.join(path, 'configurations', 'configurations.json')        
        with open(config_path, 'r') as f:
            configurations = json.load(f)        

        history_path = os.path.join(path, 'configurations', 'session_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configurations, history  

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
        custom_objects = {'MaskedSparseCategoricalCrossentropy': MaskedSparseCategoricalCrossentropy,
                          'MaskedAccuracy': MaskedAccuracy, 
                          'LRScheduler': LRScheduler}        

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name)
        model_path = os.path.join(checkpoint_path, 'saved_model.keras') 
        model = keras.models.load_model(model_path, custom_objects=custom_objects) 
        
        return model
            
    #-------------------------------------------------------------------------- 
    def select_and_load_checkpoint(self):        
        model_folders = self.scan_checkpoints_folder()
        # quit the script if no pretrained models are found 
        if len(model_folders) == 0:
            logger.error('No pretrained model checkpoints in resources')
            sys.exit()

        # select model if multiple checkpoints are available
        if len(model_folders) > 1:
            selection_index = checkpoint_selection_menu(model_folders)                    
            checkpoint_path = os.path.join(
                CHECKPOINT_PATH, model_folders[selection_index-1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            checkpoint_path = os.path.join(CHECKPOINT_PATH, model_folders[0])
            logger.info(f'Since only checkpoint {os.path.basename(checkpoint_path)} is available, it will be loaded directly')
                          
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        model = self.load_checkpoint(checkpoint_path)       
        configuration, history = self.load_session_configuration(checkpoint_path)           
            
        return model, configuration, history, checkpoint_path

             
    