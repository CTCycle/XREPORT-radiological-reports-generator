import numpy as np
import tensorflow as tf
from tensorflow import keras

    

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
    


   


    