import numpy as np
import tensorflow as tf
from tensorflow import keras

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
#------------------------------------------------------------------------------
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, shuffle=True):        
        self.dataframe = dataframe
        self.path_col='path'        
        self.label_col='tokens'
        self.batch_size = CONFIG["training"]["BATCH_SIZE"]
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]       
        self.normalization = CONFIG["dataset"]["IMG_NORMALIZE"]
        self.augmentation = CONFIG["dataset"]["IMG_AUGMENT"]         
        self.num_samples = dataframe.shape[0]         
        self.batch_index = 0              
        self.shuffle = shuffle        
        self.on_epoch_end() 
        logger.debug(f'Generator shuffle is set as {shuffle}')
        logger.debug(f'Number of samples given to generator is {self.num_samples}')      

    # define length of the custom generator      
    #--------------------------------------------------------------------------
    def __len__(self):
        length = int(np.ceil(self.num_samples)/self.batch_size)
        return length
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def __getitem__(self, idx): 
        path_batch = self.dataframe[self.path_col][idx * self.batch_size:(idx + 1) * self.batch_size]        
        label_batch = self.dataframe[self.label_col][idx * self.batch_size:(idx + 1) * self.batch_size]
        x1_batch = [self.__images_generation(image_path) for image_path in path_batch]
        x2_batch = [self.__labels_generation(label_id) for label_id in label_batch] 
        y_batch = [self.__labels_generation(label_id) for label_id in label_batch]
        X1_tensor = tf.convert_to_tensor(x1_batch)
        X2_tensor = tf.convert_to_tensor(x2_batch)
        Y_tensor = tf.convert_to_tensor(y_batch)
        X2_tensor, Y_tensor = X2_tensor[:, :-1], Y_tensor[:, 1:]

        return (X1_tensor, X2_tensor), Y_tensor
    
    # define method to perform data operations on epoch end
    #--------------------------------------------------------------------------
    def on_epoch_end(self):        
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # define method to load images and perform data augmentation    
    #--------------------------------------------------------------------------
    def __images_generation(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=1)
        pp_image = tf.image.resize(image, self.img_shape[:-1])       
        if self.augmentation:          
            pp_image = tf.keras.preprocessing.image.random_shift(pp_image, 0.2, 0.3)
            pp_image = tf.image.random_flip_left_right(pp_image)  
        if self.normalization:
            pp_image = pp_image/255.0      

        return pp_image       
    
    # define method to load labels    
    #--------------------------------------------------------------------------
    def __labels_generation(self, sequence):
        token_seq = np.fromstring(sequence, sep=' ', dtype=int)       
                 
        return token_seq
    
    # define method to call the elements of the generator    
    #--------------------------------------------------------------------------
    def next(self):
        next_index = (self.batch_index + 1) % self.__len__()
        self.batch_index = next_index
        return self.__getitem__(next_index)
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
#------------------------------------------------------------------------------
def build_tensor_dataset(dataframe, buffer_size=tf.data.AUTOTUNE):

    generator = DataGenerator(dataframe, shuffle=True)                              
        
    x_batch, y_batch = generator.__getitem__(0)   
    # ensure to replace first dimension with None for flexible batch size 
    x1_batch_shape = (None,) + x_batch[0].shape[1:]
    x2_batch_shape = (None,) + x_batch[1].shape[1:]
    y_batch_shape = (None,) + y_batch.shape[1:]      
    output_signature = ((tf.TensorSpec(shape=x1_batch_shape, dtype=tf.float32),
                        tf.TensorSpec(shape=x2_batch_shape, dtype=tf.float32)),
                        tf.TensorSpec(shape=y_batch_shape, dtype=tf.float32))        
    dataset = tf.data.Dataset.from_generator(lambda : generator, output_signature=output_signature)
    dataset = dataset.prefetch(buffer_size=buffer_size) 

    # logging debug info about batch shapes
    logger.debug(f'X batch shape is: {x1_batch_shape}')  
    logger.debug(f'Y batch shape is: {y_batch_shape}') 

    return dataset    


    


   


    