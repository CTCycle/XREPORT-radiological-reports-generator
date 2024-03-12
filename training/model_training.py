import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and classes
#------------------------------------------------------------------------------    
from utils.data_assets import PreProcessing, DataGenerator, TensorDataSet
from utils.model_assets import ModelTraining, XREPCaptioningModel
from utils.callbacks import RealTimeHistory, GenerateTextCallback
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
images_path = os.path.join(globpt.data_path, 'images') 
cp_path = os.path.join(globpt.train_path, 'checkpoints')
bert_path = os.path.join(globpt.train_path, 'BERT')
os.mkdir(images_path) if not os.path.exists(images_path) else None 
os.mkdir(cp_path) if not os.path.exists(cp_path) else None
os.mkdir(bert_path) if not os.path.exists(bert_path) else None


# [LOAD DATA]
#==============================================================================
#==============================================================================

# create model folder
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
model_folder = preprocessor.model_savefolder(cp_path, 'XREP')
model_folder_name = preprocessor.folder_name

# load data from csv, add paths to images 
#------------------------------------------------------------------------------
file_loc = os.path.join(globpt.data_path, 'XREP_dataset.csv') 
dataset = pd.read_csv(file_loc, encoding='utf-8', sep=';', low_memory=False)
dataset = preprocessor.find_images_path(images_path, dataset)

# select subset of data
#------------------------------------------------------------------------------
total_samples = cnf.num_train_samples + cnf.num_test_samples
dataset = dataset[dataset['text'].apply(lambda x: len(x.split()) <= 200)]
dataset = dataset.sample(n=total_samples, random_state=cnf.seed)

# split data into train and test dataset and start preprocessor
#------------------------------------------------------------------------------
test_size = cnf.num_test_samples/total_samples
train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=cnf.split_seed)

# [PREPROCESS DATA]
#==============================================================================
#==============================================================================

# create subfolder for preprocessing data
#------------------------------------------------------------------------------
pp_path = os.path.join(model_folder, 'preprocessing')
os.mkdir(pp_path) if not os.path.exists(pp_path) else None 

# preprocess text corpus using pretrained BioBERT tokenizer. Text is tokenized
# using subwords and these are eventually mapped to integer indexes
#------------------------------------------------------------------------------
train_text, test_text = train_data['text'].to_list(), test_data['text'].to_list()

# preprocess text with BioBERT tokenization
pad_length = max([len(x.split()) for x in train_text])
train_tokens, test_tokens = preprocessor.BERT_tokenization(train_text, test_text, bert_path)
tokenizer = preprocessor.tokenizer
vocab_size = preprocessor.vocab_size

# add tokenized text to dataframe. Sequences are converted to strings to make 
# it easy to save the files as .csv
train_ids = train_tokens['input_ids'].numpy().tolist()
test_ids = test_tokens['input_ids'].numpy().tolist()
train_data['tokens'] = [' '.join(map(str, ids)) for ids in train_ids]
test_data['tokens'] = [' '.join(map(str, ids)) for ids in test_ids]

# save preprocessed data
#------------------------------------------------------------------------------
file_loc = os.path.join(pp_path, 'XREP_train.csv')  
train_data.to_csv(file_loc, index = False, sep = ';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'XREP_test.csv')  
test_data.to_csv(file_loc, index = False, sep = ';', encoding='utf-8')

# [CREATE DATA GENERATOR]
#==============================================================================
#==============================================================================
train_data['tokens'] = train_ids
test_data['tokens'] = test_ids

# initialize training device
#------------------------------------------------------------------------------
trainer = ModelTraining(device=cnf.training_device, seed=cnf.seed)

# initialize the images generator for the train and test data, and create the 
# tf.dataset according to batch shapes
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, cnf.batch_size, cnf.picture_shape, 
                                shuffle=True, augmentation=cnf.augmentation)
test_generator = DataGenerator(test_data, cnf.batch_size, cnf.picture_shape, 
                               shuffle=True, augmentation=cnf.augmentation)

# initialize the TensorDataSet class with the generator instances
# create the tf.datasets using the previously initialized generators 
datamaker = TensorDataSet()
train_dataset = datamaker.create_tf_dataset(train_generator)
test_dataset = datamaker.create_tf_dataset(test_generator)
caption_shape = datamaker.y_batch.shape[1]

# [BUILD XREPORT MODEL]
#==============================================================================
#==============================================================================

# Print report with info about the training parameters
#------------------------------------------------------------------------------
print(f'''
-------------------------------------------------------------------------------
XRAYREP training report
-------------------------------------------------------------------------------
Number of train samples: {cnf.num_train_samples}
Number of test samples:  {cnf.num_test_samples}
-------------------------------------------------------------------------------
Batch size:              {cnf.batch_size}
Epochs:                  {cnf.epochs}
Vocabulary size:         {vocab_size + 1}
Caption length:          {caption_shape} 
-------------------------------------------------------------------------------
''')

# initialize and compile the captioning model
#------------------------------------------------------------------------------
caption_model = XREPCaptioningModel(cnf.picture_shape, caption_shape, vocab_size, 
                                    cnf.embedding_dims, cnf.kernel_size, cnf.num_heads,
                                    cnf.learning_rate, cnf.XLA_acceleration, cnf.seed)
caption_model.compile()

# invoke call method to build a showcase model (in order to show summary and plot)
#------------------------------------------------------------------------------
showcase_model = caption_model.get_model()
showcase_model.summary()

# generate graphviz plot fo the model layout
#------------------------------------------------------------------------------
if cnf.generate_model_graph == True:
    plot_path = os.path.join(model_folder, 'XREP_scheme.png')       
    plot_model(showcase_model, to_file = plot_path, show_shapes = True, 
               show_layer_names = True, show_layer_activations = True, 
               expand_nested = True, rankdir='TB', dpi = 400)    
    
# [TRAIN XREPORT MODEL]
#==============================================================================
# Setting callbacks and training routine for the XRAY captioning model. 
# to visualize tensorboard report, use command prompt on the model folder and 
# upon activating environment, use the bash command: 
# python -m tensorboard.main --logdir tensorboard/
#==============================================================================

# initialize real time plot callback 
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(model_folder, validation=True)
#GT_callback = GenerateTextCallback(tokenizer, max_len=200)

# initialize tensorboard
#------------------------------------------------------------------------------
if cnf.use_tensorboard == True:
    log_path = os.path.join(model_folder, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    callbacks = [RTH_callback, tensorboard_callback]    
else:    
    callbacks = [RTH_callback]

# define and execute training loop,
#------------------------------------------------------------------------------
multiprocessing = cnf.num_processors > 1
training = caption_model.fit(train_dataset, validation_data=test_dataset, epochs=cnf.epochs, 
                             batch_size=cnf.batch_size, callbacks=callbacks, 
                             workers=cnf.num_processors, use_multiprocessing=multiprocessing) 

# save model by saving weights and configurations, due to it being a subclassed model
# with custom train_step function 
#------------------------------------------------------------------------------
model_files_path = os.path.join(model_folder, 'model')
os.mkdir(model_files_path) if not os.path.exists(model_files_path) else None
trainer.save_subclassed_model(caption_model, model_files_path)

print(f'''
-------------------------------------------------------------------------------
Training session is over. Model has been saved in folder {model_folder_name}
-------------------------------------------------------------------------------
''')

# save model parameters in json files
#------------------------------------------------------------------------------
parameters = {'train_samples': cnf.num_train_samples,
              'test_samples': cnf.num_test_samples,
              'picture_shape' : cnf.picture_shape,             
              'kernel_size' : cnf.kernel_size, 
              'num_heads' : cnf.num_heads,             
              'augmentation' : cnf.augmentation,              
              'batch_size' : cnf.batch_size,
              'learning_rate' : cnf.learning_rate,
              'epochs' : cnf.epochs,
              'seed' : cnf.seed,
              'tensorboard' : cnf.use_tensorboard}

trainer.model_parameters(parameters, model_folder)



