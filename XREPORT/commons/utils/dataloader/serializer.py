import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
import keras
import tensorflow as tf


from XREPORT.commons.utils.models.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.utils.models.scheduler import LRScheduler
from XREPORT.commons.constants import CONFIG, DATA_PATH, CHECKPOINT_PATH, GENERATION_INPUT_PATH
from XREPORT.commons.logger import logger


# get images from the paths specified in a pandas dataframe 
###############################################################################
def get_images_from_dataset(path, dataframe, sample_size=None):     

    '''
    Maps image file paths to their corresponding entries in a dataframe.
    This function iterates over all images in a specified directory, 
    creating a mapping from image names (without the file extension) 
    to their full file paths. 

    Keyword Arguments:
        path (str): The directory containing the images.
        dataframe (pandas.DataFrame): The dataframe to be updated with image paths.

    Returns:
        pandas.DataFrame: The updated dataframe with a new 'path' column containing 
                            the file paths for images, excluding rows without a corresponding image file.
    
    ''' 
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'} 
    logger.debug(f'Valid extensions are: {valid_extensions}') 
    images_path = {}

    for root, _, files in os.walk(path):
        if sample_size is not None:
            files = files[:int(sample_size*len(files))]           
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                img_name = file.split('.')[0]  
                img_path = os.path.join(path, file)                                    
                path_pair = {img_name : img_path}        
                images_path.update(path_pair) 

    dataframe['path'] = dataframe['id'].map(images_path)
    dataframe = dataframe.dropna(subset=['path']).reset_index(drop=True)             

    return dataframe

# get the path of multiple images from a given directory
###############################################################################
def get_images_path():

    
    # check report folder and generate list of images paths    
    if not os.listdir(GENERATION_INPUT_PATH):
        logger.error('No XRAY scans found in the inference input folder, please add them before using this module!\n')
        sys.exit()
    else:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        logger.debug(f'Valid extensions are: {valid_extensions}') 
        images_path = []
        for root, _, files in os.walk(GENERATION_INPUT_PATH):                 
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    images_path.append(os.path.join(root, file))                

        return images_path
    

# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self):        
        self.color_encoding = cv2.COLOR_BGR2RGB
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]
        self.resized_img_shape = self.img_shape[:-1]
        self.normalization = CONFIG["dataset"]["IMG_NORMALIZE"]           
    
    #--------------------------------------------------------------------------
    def load_image(self, path, as_tensor=True):               
        
        if as_tensor:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, self.resized_img_shape)
            image = tf.reverse(image, axis=[-1])
            if self.normalization:
                image = image/255.0              
        else:
            image = cv2.imread(path)             
            image = cv2.resize(image, self.resized_img_shape)            
            image = cv2.cvtColor(image, self.color_encoding) 
            if self.normalization:
                image = image/255.0           

        return image

    # ...
    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, train_data : pd.DataFrame, 
                               validation_data : pd.DataFrame, path=''): 

        processing_info = {'sample_size' : CONFIG["dataset"]["SAMPLE_SIZE"],
                           'train_size' : 1.0 - - CONFIG["dataset"]["VALIDATION_SIZE"],
                           'validation_size' : CONFIG["dataset"]["VALIDATION_SIZE"],
                           'max_sequence_size' : CONFIG["dataset"]["MAX_REPORT_SIZE"],                           
                           'date': datetime.now().strftime("%Y-%m-%d")}

        # define paths of .csv and .json files
        train_pp_path = os.path.join(path, 'XREP_train.csv')
        val_pp_path = os.path.join(path, 'XREP_validation.csv')
        json_info_path = os.path.join(path, 'preprocessing_metadata.json')
        
        # save train and validation data as .csv in the dataset folder
        train_data.to_csv(train_pp_path, index=False, sep=';', encoding='utf-8')
        validation_data.to_csv(val_pp_path, index=False, sep=';', encoding='utf-8') 
        logger.debug(f'Preprocessed train data has been saved at {train_pp_path}') 
        logger.debug(f'Preprocessed validation data has been saved at {val_pp_path}') 

        # save the preprocessing info as .json file in the dataset folder
        with open(json_info_path, 'w') as file:
            json.dump(processing_info, file, indent=4) 
            logger.debug('Preprocessing info:\n%s', file)

    # ...
    #--------------------------------------------------------------------------
    def load_preprocessed_data(self):

        # load preprocessed train and validation data
        train_file_path = os.path.join(DATA_PATH, 'XREP_train.csv') 
        val_file_path = os.path.join(DATA_PATH, 'XREP_validation.csv')
        train_data = pd.read_csv(train_file_path, encoding='utf-8', sep=';', low_memory=False)
        validation_data = pd.read_csv(val_file_path, encoding='utf-8', sep=';', low_memory=False)

        # transform text strings into array of words
        train_data['tokens'] = train_data['tokens'].apply(lambda x : [int(f) for f in x.split()])
        validation_data['tokens'] = validation_data['tokens'].apply(lambda x : [int(f) for f in x.split()])
        # load preprocessing metadata
        metadata_path = os.path.join(DATA_PATH, 'preprocessing_metadata.json')
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        
        return train_data, validation_data, metadata 
    
    

# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'XREPORT'

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):

        '''
        Creates a folder with the current date and time to save the model.

        Keyword arguments:
            None

        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_folder_path = os.path.join(CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
        os.makedirs(checkpoint_folder_path, exist_ok=True)        
        os.makedirs(os.path.join(checkpoint_folder_path, 'data'), exist_ok=True)
        logger.debug(f'Created checkpoint folder at {checkpoint_folder_path}')
        
        return checkpoint_folder_path 

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model : keras.Model, path):

        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model has been saved in folder {path}')

    #--------------------------------------------------------------------------
    def save_model_parameters(self, path, parameters_dict):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            savepath (str): The directory path where the parameters will be saved.

        Returns:
            None       

        '''
        path = os.path.join(path, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f)
            logger.debug(f'Model parameters have been saved at {path}')

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):

        if CONFIG["model"]["SAVE_MODEL_PLOT"]:
            logger.debug('Generating model architecture graph')
            plot_path = os.path.join(path, 'model_layout.png')       
            keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                       show_layer_names=True, show_layer_activations=True, 
                       expand_nested=True, rankdir='TB', dpi=400)
            
    #-------------------------------------------------------------------------- 
    def load_pretrained_model(self):

        '''
        Load a pretrained Keras model from the specified directory. If multiple model 
        directories are found, the user is prompted to select one. If only one model 
        directory is found, that model is loaded directly. If a 'model_parameters.json' 
        file is present in the selected directory, the function also loads the model 
        parameters.

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                            model parameters from a JSON file. 
                                            Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.
            configuration (dict): The loaded model parameters, or None if the parameters file is not found.

        '''  
        # look into checkpoint folder to get pretrained model names      
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)

        # quit the script if no pretrained models are found 
        if len(model_folders) == 0:
            logger.error('No pretrained model checkpoints in resources')
            sys.exit()

        # select model if multiple checkpoints are available
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Currently available pretrained models:')             
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')                         
            while True:
                try:
                    dir_index = int(input('\nSelect the pretrained model: '))
                    print()
                except ValueError:
                    logger.error('Invalid choice for the pretrained model, asking again')
                    continue
                if dir_index in index_list:
                    break
                else:
                    logger.warning('Model does not exist, please select a valid index')
                    
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[dir_index - 1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            logger.info('Loading pretrained model directly as only one is available')
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[0])                 
            
        # Set dictionary of custom objects     
        custom_objects = {'MaskedSparseCategoricalCrossentropy': MaskedSparseCategoricalCrossentropy,
                          'MaskedAccuracy': MaskedAccuracy, 
                          'LRScheduler': LRScheduler}          
        
        # effectively load the model using keras builtin method
        # Load the model with the custom objects 
        model_path = os.path.join(self.loaded_model_folder, 'saved_model.keras')         
        model = keras.models.load_model(model_path, custom_objects=custom_objects)        
        
        # load configuration data from .json file in checkpoint folder
        config_path = os.path.join(self.loaded_model_folder, 'model_parameters.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configuration = json.load(f)                   
        else:
            logger.warning('model_parameters.json file not found. Model parameters were not loaded.')
            configuration = None    
            
        return model, configuration    

             
    