import os
import re
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.api._v2.keras import preprocessing
from transformers import DistilBertTokenizer


    
    
# [PREPROCESSING PIPELINE]
#==============================================================================
# Preprocess data
#==============================================================================
class PreProcessing:

    #--------------------------------------------------------------------------
    def find_images_path(self, path, dataframe):

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
        images_paths = {}
        for pic in os.listdir(path):                       
            pic_name = pic.split('.')[0]
            pic_path = os.path.join(path, pic)                        
            path_pair = {pic_name : pic_path}        
            images_paths.update(path_pair)            
        
        dataframe['path'] = dataframe['id'].map(images_paths)
        dataframe = dataframe.dropna(subset=['path']).reset_index(drop = True)

        return dataframe 

    #--------------------------------------------------------------------------
    def load_images(self, paths, image_size, as_tensor=True, normalize=True):

        '''
        Loads and preprocesses a list of images from the specified paths. Reads images 
        from given file paths, resizes them to the specified size, optionally 
        normalizes pixel values, and can return them as either raw NumPy arrays or 
        TensorFlow tensors. 

        Keyword Arguments:
            paths (list of str): File paths to the images to be loaded.
            image_size (tuple of int): The target size for resizing the images (width, height).
            as_tensor (bool, optional): If True, returns images as TensorFlow tensors; 
                                        otherwise, returns them as NumPy arrays. Default is True.
            normalize (bool, optional): If True, normalizes image pixel values to the range [0, 1]. Default is True.

        Returns:
            list: A list containing the loaded and processed images.

        '''        
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
                image = tf.image.decode_image(image, channels=1)
                image = tf.image.resize(image, image_size)
                image = tf.reverse(image, axis=[-1])
                if normalize==True:
                    image = image/255.0
            
            images.append(image) 

        return images

    #--------------------------------------------------------------------------
    def get_BERT_tokenizer(self, path):

        '''
        Loads and initializes the BioBERT Base v1.1 tokenizer. It optionally
        accepts a path to cache the tokenizer.

        Keyword Arguments:
            path (str, optional): Directory path to cache the tokenizer. 
                                  If not specified, the default cache location is used.

        Returns:
            tokenizer (AutoTokenizer): The loaded BioBERT tokenizer.

        '''
        model_identifier = 'distilbert/distilbert-base-uncased'
        print('\nLoading BERT tokenizer\n')        
        tokenizer = DistilBertTokenizer.from_pretrained(model_identifier, cache_dir=path) 

        return tokenizer      
    
    #--------------------------------------------------------------------------
    def BERT_tokenization(self, train_text, test_text=None, path=None):

        '''
        Tokenizes text data using the BioBERT Base v1.1 tokenizer. Loads the BioBERT 
        tokenizer and applies it to tokenize the provided training (and optionally testing)
        text datasets. It supports padding, truncation, and returns the tokenized data 
        in TensorFlow tensors. 

        Keyword Arguments:
            train_text (list of str): The text data for training to tokenize.
            test_text (list of str, optional): The text data for testing to tokenize. 
                                               Default is None, indicating no test text.
            path (str, optional): Path to cache the BioBERT tokenizer. 
                                  Default is None, using the default cache directory.

        Returns:
            tuple: A tuple containing two elements:
                - train_tokens (tf.Tensor): Tokenized version of `train_text`.
                - test_tokens (tf.Tensor or None): Tokenized version of `test_text` if provided, otherwise None.

        '''        
        model_identifier = 'distilbert/distilbert-base-uncased'
        print('\nLoading BERT tokenizer\n')        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_identifier, cache_dir=path)        
        train_tokens = self.tokenizer(train_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        if test_text is not None:
            test_tokens = self.tokenizer(test_text, padding=True, truncation=True, max_length=200, return_tensors='tf')
        else:
            test_tokens = None
        
        self.vocab_size = len(self.tokenizer.vocab)        
        
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
        self.path_col='path'        
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
    
     
    #--------------------------------------------------------------------------
    def create_tf_dataset(self, generator, buffer_size=tf.data.AUTOTUNE):

        '''
        Creates a TensorFlow dataset from a generator. This function initializes 
        a TensorFlow dataset using a provided generator that yields batches of 
        inputs and targets. It sets up the dataset with an appropriate output 
        signature based on the first batch of data from the generator and applies 
        prefetching to improve data loading efficiency.

        Keyword Arguments:
            generator (Generator): A generator function or an instance with a `__getitem__` method that yields batches of data.
            buffer_size (int, optional): The number of elements to prefetch in the dataset. Default is `tf.data.AUTOTUNE`, allowing TensorFlow to automatically tune the buffer size.

        Returns:
            tf.data.Dataset: A TensorFlow dataset ready for model training or evaluation.

        '''
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


    #--------------------------------------------------------------------------
    def pixel_intensity_histograms(self, image_set_1, image_set_2, path,
                                   names=['First set', 'Second set']):
        
        '''
        Generates and saves histograms of pixel intensities for two sets of images.
        This function computes the pixel intensity distributions for two sets 
        of images and plots their histograms for comparison. The histograms are 
        saved as a JPEG file in the specified path. 

        Keyword Arguments:
            image_set_1 (list of ndarray): The first set of images for histogram comparison
            image_set_2 (list of ndarray): The second set of images for histogram comparison
            path (str): Directory path where the histogram image will be saved
            names (list of str, optional): Labels for the two image sets. Default to ['First set', 'Second set']
        
        Returns:
            None
        '''
        
        pixel_intensities_1 = np.concatenate([image.flatten() for image in image_set_1])
        pixel_intensities_2 = np.concatenate([image.flatten() for image in image_set_2])        
        plt.hist(pixel_intensities_1, bins=255, alpha=0.5, density=True, color='blue', label=names[0])
        plt.hist(pixel_intensities_2, bins=255, alpha=0.7, density=True, color='green', label=names[1])
        plt.title('Pixel Intensity Histograms')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()            
        plt.tight_layout()
        plot_loc = os.path.join(path, 'pixel_intensities.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=400)
        plt.show()            
        plt.close()

    #--------------------------------------------------------------------------
    def calculate_psnr(self, img_path_1, img_path_2):
        
        img1 = cv2.imread(img_path_1)
        img2 = cv2.imread(img_path_2)      
        
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Calculate MSE
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            # The images are exactly the same
            return float('inf')
        
        # Assuming the pixel values range from 0 to 255
        PIXEL_MAX = 255.0
        
        # Calculate PSNR
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return psnr

        # Example usage
        image1_path = 'path_to_your_first_image.jpg'
        image2_path = 'path_to_your_second_image.jpg'
        psnr_value = calculate_psnr(image1_path, image2_path)
        print(f"PSNR: {psnr_value} dB")
            





if __name__ == '__main__':
    
    pp = PreProcessing()
    bio_path = os.path.join(os.getcwd(), 'training', 'BioBERT')
    
    text = ['Text example to understand the tokenizer capabilities.']
    text_tokens, _ = pp.BioBERT_tokenization(text, test_text=None, path=bio_path)
    tokenizer = pp.tokenizer    

    text_ids = text_tokens['input_ids'].numpy().tolist()
    text_tokens = tokenizer.convert_ids_to_tokens(text_tokens['input_ids'][0])

    print('Text tokens upon tokenization: ', text_tokens)
    print('Text ids upon tokenization: ', text_ids)

    cleaned_tokens = [token.replace("##", "") if token.startswith("##") else f" {token}" for token in text_tokens if token not in ['[CLS]', '[SEP]']]

    # Join the tokens to form a sentence
    sentence = ''.join(cleaned_tokens).strip()

    print(sentence)

   


    