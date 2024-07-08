import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.preprocessing.tokenizers import BERTokenizer
from XREPORT.commons.utils.dataloader.generators import build_tensor_dataset
from XREPORT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.models.training import ModelTraining
from XREPORT.commons.utils.models.captioner import XREPORTModel
from XREPORT.commons.constants import CONFIG, DATA_PATH



# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD PREPROCESSED DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images 
    train_file_path = os.path.join(DATA_PATH, 'XREP_train.csv') 
    val_file_path = os.path.join(DATA_PATH, 'XREP_validation.csv')
    train_data = pd.read_csv(train_file_path, encoding='utf-8', sep=';', low_memory=False)
    validation_data = pd.read_csv(val_file_path, encoding='utf-8', sep=';', low_memory=False)

    # create subfolder for preprocessing data    
    dataserializer = DataSerializer()
    model_folder_path = dataserializer.create_checkpoint_folder()    

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    print('\nBuilding XREPORT model and data loaders\n')     
    trainer = ModelTraining()
    trainer.set_device()

    # get tokenizers and its info
    tokenization = BERTokenizer(train_data, validation_data)    
    tokenizer = tokenization.tokenizer

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators    
    train_dataset = build_tensor_dataset(train_data)
    validation_dataset = build_tensor_dataset(validation_data)
    vocab_size = len(tokenizer.vocab) + 1

    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/

    print('\nXREPORT training report')
    print('--------------------------------------------------------------------') 
    print(f'Number of train samples:      {len(train_data)}')
    print(f'Number of validation samples: {len(validation_data)}')      
    print(f'Batch size:                   {CONFIG["training"]["BATCH_SIZE"]}')
    print(f'Epochs:                       {CONFIG["training"]["EPOCHS"]}')
    print(f'Vocabulary size:              {vocab_size}')
    print(f'Max caption length:           {CONFIG["dataset"]["MAX_CAPTION_SIZE"]}')
    print('--------------------------------------------------------------------')    

    # initialize and compile the captioning model    
    captioner = XREPORTModel(vocab_size)
    model = captioner.get_model(summary=True) 

    # generate graphviz plot fo the model layout 
    modelserializer = ModelSerializer()     
    modelserializer.save_model_plot(model, model_folder_path)              

    # perform training and save model at the end
    trainer.train_model(model, train_dataset, validation_dataset, model_folder_path)
        
    



