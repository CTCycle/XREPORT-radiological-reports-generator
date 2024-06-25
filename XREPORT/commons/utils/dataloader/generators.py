import numpy as np
import tensorflow as tf
from tensorflow import keras


from XREPORT.commons.configurations import BATCH_SIZE, IMG_SHAPE, IMG_AUGMENT
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
#------------------------------------------------------------------------------
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, shuffle=True, augmentation=True):        
        self.dataframe = dataframe
        self.path_col='path'        
        self.label_col='tokens'
        self.num_of_samples = dataframe.shape[0]         
        self.batch_index = 0              
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()       

    # define length of the custom generator      
    #--------------------------------------------------------------------------
    def __len__(self):
        length = int(np.ceil(self.num_of_samples)/BATCH_SIZE)
        return length
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def __getitem__(self, idx): 
        path_batch = self.dataframe[self.path_col][idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]        
        label_batch = self.dataframe[self.label_col][idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
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
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # define method to load images and perform data augmentation    
    #--------------------------------------------------------------------------
    def __images_generation(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=1)
        resized_image = tf.image.resize(image, IMG_SHAPE[:-1])        
        pp_image = resized_image/255.0  
        if IMG_AUGMENT:            
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
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
#------------------------------------------------------------------------------
def build_tensor_dataset(dataframe, buffer_size=tf.data.AUTOTUNE):


    generator = DataGenerator(dataframe, shuffle=True, normalization=True)                              
        
    x_batch, y_batch = generator.__getitem__(0)
    x1_shape = x_batch[0].shape
    x2_shape = x_batch[1].shape
    y_shape = y_batch.shape
    output_signature = ((tf.TensorSpec(shape=x1_shape, dtype=tf.float32),
                        tf.TensorSpec(shape=x2_shape, dtype=tf.float32)),
                        tf.TensorSpec(shape=y_shape, dtype=tf.float32))        
    dataset = tf.data.Dataset.from_generator(lambda : generator, output_signature=output_signature)
    dataset = dataset.prefetch(buffer_size=buffer_size) 

    return dataset    


    


   


    