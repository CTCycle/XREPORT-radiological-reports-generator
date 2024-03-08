import os
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.api._v2.keras import preprocessing
from transformers import AutoTokenizer, AutoModel


    
    
# [PREPROCESSING PIPELINE]
#==============================================================================
# Preprocess data
#==============================================================================
class PreProcessing:

    #--------------------------------------------------------------------------
    def images_pathfinder(self, path, dataframe, id_col):

        images_paths = {}
        for pic in os.listdir(path):
            pic_name = pic.split('.')[0]
            pic_path = os.path.join(path, pic)                        
            path_pair = {pic_name : pic_path}        
            images_paths.update(path_pair)
        
        dataframe['images_path'] = dataframe[id_col].map(images_paths)
        dataframe = dataframe.dropna(subset=['images_path']).reset_index(drop = True)

        return dataframe 

    #--------------------------------------------------------------------------
    def load_images(self, paths, image_size, as_tensor=True, normalize=True):
        
        images = []
        for pt in tqdm(paths):
            if as_tensor==False:                
                image = cv2.imread(pt)             
                image = cv2.resize(image, image_size)            
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                if normalize==True:
                    image = image/255.0
            else:
                image = tf.io.read_file(pt)
                image = tf.image.decode_image(image, channels=3)
                image = tf.image.resize(image, image_size)
                image = tf.reverse(image, axis=[-1])
                if normalize==True:
                    image = image/255.0
            
            images.append(image) 

        return images      
    
    #--------------------------------------------------------------------------
    def BioBERT_tokenization(self, train_text, test_text, path=None):

        '''        
        Tokenizes a list of texts and saves the tokenizer to a specified path.

        Keyword arguments:
            text (list): A list of texts to be tokenized.
            savepath (str): The path to save the tokenizer as a JSON file.
            
        Returns:
            tokenized_text (list or numpy.ndarray): The tokenized texts in the specified output format.
        
        '''        
        model_identifier = 'dmis-lab/biobert-base-cased-v1.1'
        print('\nLoading BioBERT Base v1.1 tokenizer\n')        
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, cache_dir=path)        
        train_tokens = tokenizer(train_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        test_tokens = tokenizer(test_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        self.vocab_size = len(tokenizer.vocab)        
        
        return train_tokens, test_tokens
 
    #--------------------------------------------------------------------------
    def sequence_padding(self, sequences, pad_length, output = 'array'):

        '''
        sequence_padding(sequences, pad_value, pad_length, output='array')

        Pads a list of sequences to a specified length with a specified value.

        Keyword arguments:
            sequences (list): A list of sequences to be padded.
            pad_value (int): The value to use for padding.
            pad_length (int): The length to pad the sequences to.
            output (str): The format of the output. If 'array', the function returns a list of 
            padded sequences as numpy arrays. If 'string', the function returns a list of padded sequences as strings.

        Returns:
            padded_text (list): A list of padded sequences in the specified output format.
        
        '''
        padded_text = preprocessing.sequence.pad_sequences(sequences, maxlen=pad_length, value=0, 
                                    dtype = 'int32', padding = 'post')
        if output == 'string':
            padded_text_str = []
            for x in padded_text:
                x_string = ' '.join(str(i) for i in x)
                padded_text_str.append(x_string)
            padded_text = padded_text_str          
        
        return padded_text   

    #--------------------------------------------------------------------------
    def load_tokenizer(self, path, filename):  

        json_path = os.path.join(path, f'{filename}.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            json_string = f.read()
            tokenizer = preprocessing.text.tokenizer_from_json(json_string)

        return tokenizer
    
    #--------------------------------------------------------------------------
    def model_savefolder(self, path, model_name):

        '''
        Creates a folder with the current date and time to save the model.
    
        Keyword arguments:
            path (str):       A string containing the path where the folder will be created.
            model_name (str): A string containing the name of the model.
    
        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        today_datetime = str(datetime.now())
        truncated_datetime = today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
        self.folder_name = f'{model_name}_{today_datetime}'
        model_folder_path = os.path.join(path, self.folder_name)
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path) 
                    
        return model_folder_path
    
    
# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
# Generate data on the fly to avoid memory burdening
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size=6, picture_size=(244, 244, 1), 
                 shuffle=True, augmentation=True):        
        self.dataframe = dataframe
        self.path_col='images_path'        
        self.label_col='tokens'
        self.num_of_samples = dataframe.shape[0]        
        self.picture_size = picture_size       
        self.batch_size = batch_size  
        self.batch_index = 0              
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()       

    # define length of the custom generator      
    #--------------------------------------------------------------------------
    def __len__(self):
        length = int(np.ceil(self.num_of_samples)/self.batch_size)
        return length
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def __getitem__(self, idx): 
        path_batch = self.dataframe[self.path_col][idx * self.batch_size:(idx + 1) * self.batch_size]        
        label_batch = self.dataframe[self.label_col][idx * self.batch_size:(idx + 1) * self.batch_size]
        x1_batch = [self.__images_generation(image_path, self.augmentation) for image_path in path_batch]
        x2_batch = [self.__labels_generation(label_id) for label_id in label_batch] 
        y_batch = [self.__labels_generation(label_id) for label_id in label_batch]
        X1_tensor = tf.convert_to_tensor(x1_batch)
        X2_tensor = tf.convert_to_tensor(x2_batch)
        Y_tensor = tf.convert_to_tensor(y_batch)
        return (X1_tensor, X2_tensor), Y_tensor
    
    # define method to perform data operations on epoch end
    #--------------------------------------------------------------------------
    def on_epoch_end(self):        
        self.indexes = np.arange(self.num_of_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # define method to load images and perform data augmentation    
    #--------------------------------------------------------------------------
    def __images_generation(self, path, augmentation=False):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=1)
        resized_image = tf.image.resize(image, self.picture_size[:-1])        
        pp_image = resized_image/255.0  
        if augmentation==True:            
            pp_image = tf.keras.preprocessing.image.random_shift(pp_image, 0.2, 0.3)
            pp_image = tf.image.random_flip_left_right(pp_image)        

        return pp_image       
    
    # define method to load labels    
    #--------------------------------------------------------------------------
    def __labels_generation(self, sequence):        
        return sequence
    
    # define method to call the elements of the generator    
    #--------------------------------------------------------------------------
    def next(self):
        next_index = (self.batch_index + 1) % self.__len__()
        self.batch_index = next_index
        return self.__getitem__(next_index)
    

# [TF.DATASET GENERATION]
#==============================================================================
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
#==============================================================================
class TensorDataSet():

    
    # create tensorflow dataset from generator    
    #--------------------------------------------------------------------------
    def create_tf_dataset(self, generator, buffer_size=tf.data.AUTOTUNE):

        self.x_batch, self.y_batch = generator.__getitem__(0)
        x1_shape = self.x_batch[0].shape
        x2_shape = self.x_batch[1].shape
        y_shape = self.y_batch.shape
        output_signature = ((tf.TensorSpec(shape=x1_shape, dtype=tf.float32),
                            tf.TensorSpec(shape=x2_shape, dtype=tf.float32)),
                            tf.TensorSpec(shape=y_shape, dtype=tf.float32))        
        dataset = tf.data.Dataset.from_generator(lambda : generator, output_signature=output_signature)
        dataset = dataset.prefetch(buffer_size=buffer_size) 

        return dataset
    

# [VALIDATION OF DATA]
#==============================================================================
# Series of methods and functions to preprocess data for model training
#==============================================================================
class DataValidation:

    def pixel_intensity_histograms(self, image_set_1, image_set_2, path,
                                   names=['First set', 'Second set']):
        
        pixel_intensities_1 = np.concatenate([image.flatten() for image in image_set_1])
        pixel_intensities_2 = np.concatenate([image.flatten() for image in image_set_2])        
        plt.hist(pixel_intensities_1, bins='auto', alpha=0.5, color='blue', label=names[0])
        plt.hist(pixel_intensities_2, bins='auto', alpha=0.5, color='red', label=names[1])
        plt.title('Pixel Intensity Histograms')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()            
        plt.tight_layout()
        plot_loc = os.path.join(path, 'pixel_intensities.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=400)
        plt.show()            
        plt.close()
        





if __name__ == '__main__':
    
    pp = PreProcessing()
    

      
