import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from keras.api._v2.keras import preprocessing
from transformers import DistilBertTokenizer

    
#------------------------------------------------------------------------------
def find_images_path(path, dataframe):

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

#------------------------------------------------------------------------------
def load_images(paths, image_size, as_tensor=True, normalize=True):

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


# [PREPROCESSING PIPELINE]
#==============================================================================
# Preprocess data
#==============================================================================
class PreProcessing:    

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
    
   
