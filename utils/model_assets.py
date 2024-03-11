import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras import layers
from transformers import TFAutoModelForMaskedLM
    
           
# [LEARNING RATE SCHEDULER]
#==============================================================================
# Generate data on the fly to avoid memory burdening
#==============================================================================
@keras.utils.register_keras_serializable(package='LRScheduler')
class LRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_lr, warmup_steps):
        super().__init__()
        self.post_warmup_lr = post_warmup_lr
        self.warmup_steps = warmup_steps

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step/warmup_steps
        warmup_learning_rate = self.post_warmup_lr * warmup_progress

        return tf.cond(global_step < warmup_steps, lambda: warmup_learning_rate,
                       lambda: self.post_warmup_lr)
    
    # custom configurations
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(LRScheduler, self).get_config()
        config.update({'post_warmup_lr': self.post_warmup_lr,
                       'warmup_steps': self.warmup_steps})
        return config        
    
    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
      
# [IMAGE ENCODER MODEL]
#==============================================================================
# Custom encoder model
#==============================================================================    
@keras.utils.register_keras_serializable(package='Encoders', name='ImageEncoder')
class ImageEncoder(keras.layers.Layer):
    def __init__(self, kernel_size, seed, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.seed = seed
        self.conv1 = layers.Conv2D(64, kernel_size, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.conv2 = layers.Conv2D(128, kernel_size, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.conv3 = layers.Conv2D(256, kernel_size, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')  
        self.conv4 = layers.Conv2D(256, kernel_size, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.conv5 = layers.Conv2D(512, kernel_size, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform') 
        self.conv6 = layers.Conv2D(512, kernel_size, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.maxpool1 = layers.MaxPooling2D((2, 2), strides=2, padding='same')
        self.maxpool2 = layers.MaxPooling2D((2, 2), strides=2, padding='same')
        self.maxpool3 = layers.MaxPooling2D((2, 2), strides=2, padding='same')
        self.maxpool4 = layers.MaxPooling2D((2, 2), strides=2, padding='same')          
        self.dense1 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dense3 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')
        self.reshape = layers.Reshape((-1, 128))        

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, x, training=True):              
        layer = self.conv1(x)                  
        layer = self.maxpool1(layer) 
        layer = self.conv2(layer)                     
        layer = self.maxpool2(layer)        
        layer = self.conv3(layer)  
        layer = self.conv4(layer)                        
        layer = self.maxpool3(layer)                
        layer = self.conv5(layer) 
        layer = self.conv6(layer)               
        layer = self.maxpool4(layer)         
        layer = self.dense1(layer)        
        layer = self.dense2(layer)       
        layer = self.dense3(layer)       
        output = self.reshape(layer)              
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(ImageEncoder, self).get_config()       
        config.update({'kernel_size': self.kernel_size,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [POSITIONAL EMBEDDING]
#==============================================================================
# Custom positional embedding layer
#==============================================================================
@keras.utils.register_keras_serializable(package='CustomLayers', name='PositionalEmbedding')
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embedding_dims, bio_path, mask_zero=True, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size 
        self.embedding_dim = embedding_dims
        self.bio_path = bio_path
        self.mask_zero = mask_zero

        # token embedding using BioBERT, embedding dims is 768
        model_identifier = 'emilyalsentzer/Bio_ClinicalBERT'
        biobert_model = TFAutoModelForMaskedLM.from_pretrained(model_identifier, from_pt=True, cache_dir=bio_path) 
        self.token_embeddings = biobert_model.get_input_embeddings() # Extract embedding layer  
        self.token_embeddings.trainable = True                         
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embedding_dims)        
        self.embedding_scale = tf.math.sqrt(tf.cast(embedding_dims, tf.float32))        
    
    # implement positional embedding through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)        
        embedded_tokens = self.token_embeddings(inputs)  
        embedded_tokens = embedded_tokens * self.embedding_scale
        embedded_positions = self.position_embeddings(positions)
        full_embedding = embedded_tokens + embedded_positions        
        if self.mask_zero==True:
            mask = tf.math.not_equal(inputs, 0)
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)                              
            full_embedding = full_embedding * mask            

        return full_embedding
    
    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({'sequence_length': self.sequence_length,
                       'vocab_size': self.vocab_size,
                       'embedding_dims': self.embedding_dim,
                       'bio_path' : self.bio_path,
                       'mask_zero': self.mask_zero})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [TRANSFORMER ENCODER]
#==============================================================================
# Custom transformer encoder
#============================================================================== 
@keras.utils.register_keras_serializable(package='Encoders', name='TransformerEncoderBlock')
class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads, seed, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims       
        self.num_heads = num_heads  
        self.seed = seed       
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dense1 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dense3 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dense4 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')
        self.dropout1 = layers.Dropout(0.2, seed=seed)
        self.dropout2 = layers.Dropout(0.3, seed=seed)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):        
        inputs = self.layernorm1(inputs)
        inputs = self.dense1(inputs)
        inputs = self.dropout1(inputs)  
        inputs = self.dense2(inputs)            
        attention_output = self.attention(query=inputs, value=inputs, key=inputs,
                                          attention_mask=None, training=training)        
        layernorm = self.layernorm2(inputs + attention_output)
        layer = self.dense3(layernorm)
        layer = self.dropout2(layer)
        output = self.dense4(layer)        

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({'embedding_dims': self.embedding_dims,
                       'num_heads': self.num_heads,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [TRANSFORMER DECODER]
#==============================================================================
# Custom transformer decoder
#============================================================================== 
@keras.utils.register_keras_serializable(package='Decoders', name='TransformerDecoderBlock')
class TransformerDecoderBlock(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embedding_dims, bio_path, num_heads, seed, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims 
        self.bio_path = bio_path       
        self.num_heads = num_heads
        self.seed = seed        
        self.posembedding = PositionalEmbedding(sequence_length, vocab_size, embedding_dims, bio_path, mask_zero=True)          
        self.MHA_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims, dropout=0.2)
        self.MHA_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims, dropout=0.2)
        self.FFN_1 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.FFN_2 = layers.Dense(self.embedding_dims, activation='relu', kernel_initializer='he_uniform')
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()
        self.dense = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')         
        self.outmax = layers.Dense(self.vocab_size, activation='softmax')
        self.dropout1 = layers.Dropout(0.2, seed=seed)
        self.dropout2 = layers.Dropout(0.3, seed=seed) 
        self.dropout3 = layers.Dropout(0.3, seed=seed)
        self.supports_masking = True 

    # implement transformer decoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, encoder_outputs, training=True, mask=None):
        inputs = tf.cast(inputs, tf.int32)
        inputs = self.posembedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)        
        padding_mask = None
        combined_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)           
        attention_output1 = self.MHA_1(query=inputs, value=inputs, key=inputs,
                                       attention_mask=combined_mask, training=training)
        output1 = self.layernorm1(inputs + attention_output1)                       
        attention_output2 = self.MHA_2(query=output1, value=encoder_outputs,
                                       key=encoder_outputs, attention_mask=padding_mask,
                                       training=training)
        output2 = self.layernorm2(output1 + attention_output2)
        ffn_out = self.FFN_1(output2)
        ffn_out = self.dropout1(ffn_out, training=training)
        ffn_out = self.FFN_2(ffn_out)
        ffn_out = self.layernorm3(ffn_out + output2, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)         
        ffn_out = self.dense(ffn_out)   
        ffn_out = self.dropout3(ffn_out, training=training)     
        preds = self.outmax(ffn_out)

        return preds

    # generate causal attention mask   
    #--------------------------------------------------------------------------
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
                          axis=0)
        
        return tf.tile(mask, mult) 
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransformerDecoderBlock, self).get_config()
        config.update({'sequence_length': self.sequence_length,
                       'vocab_size': self.vocab_size,
                       'embedding_dims': self.embedding_dims,
                       'bio_path' : self.bio_path,
                       'num_heads': self.num_heads,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     

# [XREP CAPTIONING MODEL]
#==============================================================================
# Custom captioning model
#==============================================================================  
@keras.utils.register_keras_serializable(package='Models', name='XREPCaptioningModel')
class XREPCaptioningModel(keras.Model):    
    def __init__(self, picture_shape, sequence_length, vocab_size, embedding_dims, bio_path, kernel_size,
                 num_heads, learning_rate, XLA_state, seed=42, **kwargs):   
        super(XREPCaptioningModel, self).__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.acc_tracker = keras.metrics.Mean(name='accuracy')
        self.picture_shape = picture_shape
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size 
        self.embedding_dims = embedding_dims
        self.bio_path = bio_path
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.XLA_state = XLA_state                
        self.seed = seed                         
        self.image_encoder = ImageEncoder(kernel_size, seed)        
        self.encoder1 = TransformerEncoderBlock(embedding_dims, num_heads, seed)
        self.encoder2 = TransformerEncoderBlock(embedding_dims, num_heads, seed)
        self.decoder = TransformerDecoderBlock(sequence_length, self.vocab_size, embedding_dims, 
                                               bio_path, num_heads, seed) 

    # calculate loss
    #--------------------------------------------------------------------------
    def calculate_loss(self, y_true, y_pred, mask):               
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss)/(tf.reduce_sum(mask) + keras.backend.epsilon())
    
    # calculate accuracy
    #--------------------------------------------------------------------------
    def calculate_accuracy(self, y_true, y_pred, mask): 
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred_argmax = tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.float32)
        accuracy = tf.equal(y_true, y_pred_argmax)
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracy)/(tf.reduce_sum(mask) + keras.backend.epsilon())

    # calculate the caption loss and accuracy
    #--------------------------------------------------------------------------
    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):        
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        encoder_out = self.encoder1(img_embed, training=training)
        encoder_out = self.encoder2(encoder_out, training=training)         
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

        return loss, acc
    
    # define train step
    #--------------------------------------------------------------------------
    def train_step(self, batch_data):
        x_data, y_data = batch_data
        batch_img, batch_seq = x_data        
        img_embed = self.image_encoder(batch_img)       
        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=True)        
        train_vars = self.encoder1.trainable_variables + self.encoder2.trainable_variables + self.decoder.trainable_variables        
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))        
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)              
        
        return {'loss': self.loss_tracker.result(),
                'acc': self.acc_tracker.result()}

    # define test step
    #--------------------------------------------------------------------------
    def test_step(self, batch_data):
        x_data, y_data = batch_data
        batch_img, batch_seq = x_data         
        img_embed = self.image_encoder(batch_img)        
        loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=False)         
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)        

        return {'loss': self.loss_tracker.result(),
                'acc': self.acc_tracker.result()}        
 
    # implement captioning model through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=True):

        images, sequences = inputs        
        mask = tf.math.not_equal(sequences, 0)             
        image_features = self.image_encoder(images)        
        encoder = self.encoder1(image_features, training=training)
        encoder = self.encoder2(encoder, training)
        decoder = self.decoder(sequences, encoder, training=training, mask=mask)

        return decoder   
    
    # print summary
    #--------------------------------------------------------------------------
    def get_model(self):
        image_input = layers.Input(shape=self.picture_shape)    
        seq_input = layers.Input(shape=(self.sequence_length, ))
        model = Model(inputs=[image_input, seq_input], outputs = self.call([image_input, seq_input], 
                      training=False)) 
        
        return model       

    # compile the model
    #--------------------------------------------------------------------------
    def compile(self):
        lr_schedule = LRScheduler(self.learning_rate, warmup_steps=10)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, 
                                                          reduction=keras.losses.Reduction.NONE)  
        metric = keras.metrics.SparseCategoricalAccuracy()  
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)          
        super(XREPCaptioningModel, self).compile(optimizer=opt, loss=loss, metrics=metric, 
                                                 run_eagerly=False, jit_compile=self.XLA_state)   
        
    # track metrics and losses  
    #--------------------------------------------------------------------------
    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]     
 
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(XREPCaptioningModel, self).get_config()
        config.update({'picture_shape': self.picture_shape,
                       'sequence_length': self.sequence_length,
                       'vocab_size': self.vocab_size,
                       'embedding_dims': self.embedding_dims,
                       'bio_path' : self.bio_path,
                       'kernel_size': self.kernel_size,
                       'num_heads': self.num_heads,
                       'learning_rate' : self.learning_rate,
                       'XLA_state' : self.XLA_state,                 
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    

# [TRAINING OPTIONS]
#==============================================================================
# Custom training operations
#==============================================================================
class ModelTraining:

    '''
    ModelTraining - A class for configuring the device and settings for model training.

    Keyword Arguments:
        device (str):                         The device to be used for training. 
                                              Should be one of ['default', 'GPU', 'CPU'].
                                              Defaults to 'default'.
        seed (int, optional):                 The seed for random initialization. Defaults to 42.
        use_mixed_precision (bool, optional): Whether to use mixed precision for improved training performance.
                                              Defaults to False.
    
    '''      
    def __init__(self, device='default', seed=42, use_mixed_precision=False):                            
        np.random.seed(seed)
        tf.random.set_seed(seed)         
        self.available_devices = tf.config.list_physical_devices()
        print('-------------------------------------------------------------------------------')        
        print('The current devices are available: ')
        print('-------------------------------------------------------------------------------')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('-------------------------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                print('No GPU found. Falling back to CPU')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if use_mixed_precision == True:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
                os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'                 
                print('GPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()         
    
    #-------------------------------------------------------------------------- 
    def model_parameters(self, parameters_dict, savepath):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            savepath (str): The directory path where the parameters will be saved.

        Returns:
            None       

        '''
        path = os.path.join(savepath, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f)     
    
    #--------------------------------------------------------------------------   
    def save_subclassed_model(self, model, path):

        '''
        Saves a subclassed Keras model's weights and configuration to the specified directory.

        Keyword Arguments:
            model (keras.Model): The model to save.
            path (str): Directory path for saving model weights and configuration.        

        Returns:
            None
        '''        
        weights_path = os.path.join(path, 'model_weights.h5')  
        model.save_weights(weights_path)        
        config = model.get_config()
        config_path = os.path.join(path, 'model_configuration.json')
        with open(config_path, 'w') as json_file:
            json.dump(config, json_file)
        config_path = os.path.join(path, 'model_architecture.json')
        with open(config_path, 'w') as json_file:
            json_file.write(model.to_json())             
    
    
# [TOOLKIT TO USE THE PRETRAINED MODEL]
#==============================================================================
# Custom training operations
#==============================================================================
class Inference:

    def __init__(self, seed):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)  
   
    #--------------------------------------------------------------------------
    def load_pretrained_model(self, path):

        '''
        Load pretrained keras model (in folders) from the specified directory. 
        If multiple model directories are found, the user is prompted to select one,
        while if only one model directory is found, that model is loaded directly.
        If `load_parameters` is True, the function also loads the model parameters 
        from the target .json file in the same directory. 

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                              model parameters from a JSON file. 
                                              Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.

        '''        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Please select a pretrained model:') 
            print()
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')        
            print()               
            while True:
                try:
                    dir_index = int(input('Type the model index to select it: '))
                    print()
                except:
                    continue
                break                         
            while dir_index not in index_list:
                try:
                    dir_index = int(input('Input is not valid! Try again: '))
                    print()
                except:
                    continue
            self.folder_path = os.path.join(path, model_folders[dir_index - 1])

        elif len(model_folders) == 1:
            self.folder_path = os.path.join(path, model_folders[0])                   
        
        # read model serialization configuration and initialize it           
        path = os.path.join(self.folder_path, 'model', 'model_configuration.json')
        with open(path, 'r') as f:
            configuration = json.load(f)        
        model = XREPCaptioningModel.from_config(configuration)             

        # set inputs to build the model 
        pic_shape = tuple(configuration['picture_shape'])
        sequence_length = configuration['sequence_length']
        build_inputs = (tf.constant(0.0, shape=(1, *pic_shape)),
                        tf.constant(0, shape=(1, sequence_length), dtype=tf.int32))
        model(build_inputs, training=False) 

        # load weights into the model 
        weights_path = os.path.join(self.folder_path, 'model', 'model_weights.h5')
        model.load_weights(weights_path)                       
        
        return model, configuration   
                   

    #--------------------------------------------------------------------------    
    def generate_reports(self, model, paths, picture_size, max_length, tokenizer):
        
        reports = {}
        vocabulary = tokenizer.get_vocab()
        start_token = '[CLS]'
        end_token = '[SEP]'        
        index_lookup = dict(zip(range(len(vocabulary)), vocabulary))
        for pt in paths:
            print(f'\nGenerating report for images {os.path.basename(pt)}\n')
            image = tf.io.read_file(pt)
            image = tf.image.decode_image(image, channels=1)
            image = tf.image.resize(image, picture_size)            
            image = image/255.0 
            input_image = tf.expand_dims(image, 0)
            features = model.image_encoder(input_image)
            encoded_img = model.encoder1(features, training=False)   

            # teacher forging method to generate tokens through the decoder
            decoded_caption = [start_token]                 
            for i in range(max_length):     
                tokenized_caption = tokenizer.convert_tokens_to_ids(decoded_caption)                                                       
                tokenized_caption = tf.constant(tokenized_caption, dtype=tf.int32)                   
                tokenized_caption = tf.reshape(tokenized_caption, (1, -1))                                    
                mask = tf.math.not_equal(tokenized_caption, 0)                                
                predictions = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)                                                         
                sampled_token_index = np.argmax(predictions[0, i, :])                
                sampled_token = index_lookup[sampled_token_index]                                               
                if sampled_token == end_token: 
                    break
                decoded_caption.append(sampled_token)                            

            cleaned_caption = [token.replace("##", "") if token.startswith("##") else f" {token}" for token in decoded_caption if token not in ['[CLS]', '[SEP]']]
            caption = ''.join(cleaned_caption)
            reports[f'{os.path.basename(pt)}'] = caption
            print(f'Predicted report for image: {os.path.basename(pt)}', caption)          

        return reports   



    
# [VALIDATION OF PRETRAINED MODELS]
#==============================================================================
# Validation and evaluation of model performances
#==============================================================================
class ModelValidation:

    def __init__(self, model):
        
        self.model = model

    # model validation
    #-------------------------------------------------------------------------- 
    def XREPORT_validation(self, real_images, predicted_images, path):
        
        pass



if __name__ == '__main__':


    bio_path = os.path.join(os.getcwd(), 'training', 'BioBERT')
    print(bio_path)


    model_identifier = 'dmis-lab/biobert-base-cased-v1.1'
    biobert_model = TFAutoModelForMaskedLM.from_pretrained(model_identifier, from_pt=True, cache_dir=bio_path) 
    token_embeddings = biobert_model.get_input_embeddings() # Extract embedding layer  
    token_embeddings.trainable = True