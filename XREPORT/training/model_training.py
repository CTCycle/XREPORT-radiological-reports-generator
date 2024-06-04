import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.utils.preprocessing import PreProcessing, find_images_path
from XREPORT.utils.generators import DataGenerator, TensorDataSet
from XREPORT.utils.models import ModelTraining, XREPCaptioningModel, ModelValidation, model_savefolder, save_model_parameters
from XREPORT.utils.callbacks import RealTimeHistory, GenerateTextCallback
from XREPORT.config.pathfinder import IMG_DATA_PATH, DATA_PATH, CHECKPOINT_PATH, BERT_PATH
import XREPORT.config.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------     
    preprocessor = PreProcessing()   
    model_folder, model_folder_name = model_savefolder(CHECKPOINT_PATH, 'XREP')

    # load data from csv, add paths to images     
    file_loc = os.path.join(DATA_PATH, 'XREP_dataset.csv') 
    dataset = pd.read_csv(file_loc, encoding='utf-8', sep=';', low_memory=False)
    dataset = find_images_path(IMG_DATA_PATH, dataset)

    # select subset of data    
    total_samples = cnf.TRAIN_SAMPLES + cnf.TEST_SAMPLES
    dataset = dataset[dataset['text'].apply(lambda x: len(x.split()) <= 200)]
    dataset = dataset.sample(n=total_samples, random_state=cnf.SEED)

    # split data into train and test dataset and start preprocessor    
    test_size = cnf.TEST_SAMPLES/total_samples
    train_data, test_data = train_test_split(dataset, test_size=test_size, 
                                             random_state=cnf.SPLIT_SEED)

    # 2. [PREPROCESS DATA]
    #-------------------------------------------------------------------------- 

    # create subfolder for preprocessing data    
    pp_path = os.path.join(model_folder, 'preprocessing')
    os.mkdir(pp_path) if not os.path.exists(pp_path) else None 

    # preprocess text corpus using pretrained BPE tokenizer. Text is tokenized
    # using subwords and these are eventually mapped to integer indexes    
    train_text, test_text = train_data['text'].to_list(), test_data['text'].to_list()

    # preprocess text with BERT tokenization
    pad_length = max([len(x.split()) for x in train_text])
    train_tokens, test_tokens = preprocessor.BERT_tokenization(train_text, test_text, BERT_PATH)
    tokenizer = preprocessor.tokenizer
    vocab_size = preprocessor.vocab_size

    # add tokenized text to dataframe. Sequences are converted to strings to make 
    # it easy to save the files as .csv
    train_ids = train_tokens['input_ids'].numpy().tolist()
    test_ids = test_tokens['input_ids'].numpy().tolist()
    train_data['tokens'] = [' '.join(map(str, ids)) for ids in train_ids]
    test_data['tokens'] = [' '.join(map(str, ids)) for ids in test_ids]

    # save preprocessed data    
    file_loc = os.path.join(pp_path, 'XREP_train.csv')  
    train_data.to_csv(file_loc, index=False, sep =';', encoding='utf-8')
    file_loc = os.path.join(pp_path, 'XREP_test.csv')  
    test_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')

    # 3. [CREATE DATA GENERATOR]
    #-------------------------------------------------------------------------- 
    train_data['tokens'] = train_ids
    test_data['tokens'] = test_ids
    
    trainer = ModelTraining(seed=cnf.SEED)
    trainer.set_device(device=cnf.ML_DEVICE, use_mixed_precision=cnf.MIXED_PRECISION)

    # initialize the images generator for the train and test data, and create the 
    # tf.dataset according to batch shapes    
    train_generator = DataGenerator(train_data, cnf.BATCH_SIZE, cnf.IMG_SHAPE, 
                                    shuffle=True, augmentation=cnf.IMG_AUGMENT)
    test_generator = DataGenerator(test_data, cnf.BATCH_SIZE, cnf.IMG_SHAPE, 
                                shuffle=True, augmentation=cnf.IMG_AUGMENT)

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators 
    datamaker = TensorDataSet()
    train_dataset = datamaker.create_tf_dataset(train_generator)
    test_dataset = datamaker.create_tf_dataset(test_generator)
    caption_shape = datamaker.y_batch.shape[1]

    # 4. [BUILD XREPORT MODEL]
    #--------------------------------------------------------------------------    
    print('XRAYREP training report\n')
    print(f'Number of train samples: {cnf.TRAIN_SAMPLES}')
    print(f'Number of test samples:  {cnf.TEST_SAMPLES}')   
    print(f'Batch size:              {cnf.BATCH_SIZE}')
    print(f'Epochs:                  {cnf.EPOCHS}')
    print(f'Vocabulary size:         {vocab_size + 1}')
    print(f'Caption length:          {caption_shape}')
   

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




