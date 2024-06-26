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
from XREPORT.commons.utils.dataloader.serializer import get_images_from_dataset, DataSerializer, ModelSerializer

from XREPORT.commons.utils.models.training import ModelTraining
from XREPORT.commons.utils.models.captioner import XREPCaptioningModel

from XREPORT.commons.utils.models.callbacks import RealTimeHistory, GenerateTextCallback
from XREPORT.commons.pathfinder import TABLES_PATH, BERT_PATH
from XREPORT.commons.configurations import MAX_CAPTION_SIZE, BATCH_SIZE, EPOCHS


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD PREPROCESSED DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images 
    train_file_path = os.path.join(TABLES_PATH, 'XREP_train.csv') 
    val_file_path = os.path.join(TABLES_PATH, 'XREP_validation.csv')
    train_data = pd.read_csv(train_file_path, encoding = 'utf-8', sep =';', low_memory=False)
    validation_data = pd.read_csv(val_file_path, encoding = 'utf-8', sep =';', low_memory=False)

    # create subfolder for preprocessing data    
    dataserializer = DataSerializer()
    model_folder_path = dataserializer.create_checkpoint_folder()    

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    print('Building XREPORT model and data loaders\n')     
    trainer = ModelTraining()
    trainer.set_device()

    # get tokenizers and its info
    tokenization = BERTokenizer(train_data, validation_data, path=BERT_PATH)    
    tokenizer = tokenization.get_BERT_tokenizer()

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators    
    train_dataset = build_tensor_dataset(train_data)
    validation_dataset = build_tensor_dataset(validation_data)

    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/

    print('\nXREPORT training report')
    print('--------------------------------------------------------------------') 
    print(f'Number of train samples:      {len(train_data)}')
    print(f'Number of validation samples: {len(validation_data)}')      
    print(f'Batch size:                   {BATCH_SIZE}')
    print(f'Epochs:                       {EPOCHS}')
    print(f'Vocabulary size:              {len(tokenizer.vocab) + 1}')
    print(f'Max caption length:           {MAX_CAPTION_SIZE}')
    print('--------------------------------------------------------------------') 
   

    # initialize and compile the captioning model    
    caption_model = XREPCaptioningModel(cnf.IMG_SHAPE, caption_shape, vocab_size, 
                                        cnf.EMBEDDING_DIMS, cnf.KERNEL_SIZE, cnf.NUM_HEADS,
                                        cnf.LEARNING_RATE, cnf.XLA_STATE, cnf.SEED)
    caption_model.compile()

    # invoke call method to build a showcase model (in order to show summary and plot)    
    showcase_model = caption_model.get_model()
    showcase_model.summary()

    # generate graphviz plot fo the model layout    
    if cnf.SAVE_MODEL_PLOT == True:
        plot_path = os.path.join(model_folder, 'XREP_scheme.png')       
        plot_model(showcase_model, to_file = plot_path, show_shapes = True, 
                show_layer_names = True, show_layer_activations = True, 
                expand_nested = True, rankdir='TB', dpi = 400)    
        
    # 5. [TRAIN XREPORT MODEL]
    #-------------------------------------------------------------------------- 
    # Setting callbacks and training routine for the XRAY captioning model. 
    # to visualize tensorboard report, use command prompt on the model folder and 
    # upon activating environment, use the bash command: 
    # python -m tensorboard.main --logdir tensorboard/    

    # initialize real time plot callback    
    RTH_callback = RealTimeHistory(model_folder, validation=True)
    #GT_callback = GenerateTextCallback(tokenizer, max_len=200)

    # initialize tensorboard    
    if cnf.USE_TENSORBOARD == True:
        log_path = os.path.join(model_folder, 'tensorboard')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
        callbacks = [RTH_callback, tensorboard_callback]    
    else:    
        callbacks = [RTH_callback]

    # define and execute training loop   
    multiprocessing = cnf.NUM_PROCESSORS > 1
    training = caption_model.fit(train_dataset, validation_data=test_dataset, epochs=cnf.EPOCHS, 
                                batch_size=cnf.BATCH_SIZE, callbacks=callbacks, 
                                workers=cnf.NUM_PROCESSORS, use_multiprocessing=multiprocessing) 

    # save model by saving weights and configurations, due to it being a subclassed model
    # with custom train_step function     
    model_files_path = os.path.join(model_folder, 'model')
    os.mkdir(model_files_path) if not os.path.exists(model_files_path) else None
    trainer.save_subclassed_model(caption_model, model_files_path)

    print(f'\nTraining session is over. Model has been saved in folder {model_folder_name}')

    # save model parameters in json files    
    parameters = {'train_samples': cnf.TRAIN_SAMPLES,
                'test_samples': cnf.TEST_SAMPLES,
                'picture_shape' : cnf.IMG_SHAPE,                           
                'augmentation' : cnf.IMG_AUGMENT,              
                'batch_size' : cnf.BATCH_SIZE,
                'learning_rate' : cnf.LEARNING_RATE,
                'epochs' : cnf.EPOCHS,
                'seed' : cnf.SEED,
                'tensorboard' : cnf.USE_TENSORBOARD}

    save_model_parameters(parameters, model_folder) 

    # check model weights    
    validator = ModelValidation()
    validator.model_weigths_check(caption_model, model_folder)




