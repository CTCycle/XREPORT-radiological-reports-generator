import os
import re
import numpy as np
import tensorflow as tf    

from XREPORT.commons.utils.dataloader.serializer import DataSerializer
from XREPORT.commons.constants import CONFIG



# [TOOLKIT TO USE THE PRETRAINED MODEL]
#------------------------------------------------------------------------------
class TextGenerator:

    def __init__(self, model):
        
        np.random.seed(CONFIG["SEED"])
        tf.random.set_seed(CONFIG["SEED"])
        self.dataserializer = DataSerializer()
        self.model = model   
    
    #--------------------------------------------------------------------------    
    def greed_search_generator(self, model, paths, picture_size, max_length, tokenizer):
        
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
            encoded_img = model.layers[1](features, training=False)   
            encoded_img = model.layers[2](encoded_img, training=False)  
            encoded_img = model.layers[3](encoded_img, training=False)  

            # teacher forging method to generate tokens through the decoder
            decoded_caption = start_token              
            for i in range(max_length):                     
                tokenized_outputs = tokenizer(decoded_caption, add_special_tokens=False, return_tensors='tf',
                                              padding='max_length', max_length=max_length) 
                  
                tokenized_caption = tokenized_outputs['input_ids']                                                                                                        
                tokenized_caption = tf.constant(tokenized_caption, dtype=tf.int32)                                                      
                mask = tf.math.not_equal(tokenized_caption, 0)                                                
                predictions = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)                                                                                                                    
                sampled_token_index = np.argmax(predictions[0, i, :])                               
                sampled_token = index_lookup[sampled_token_index]                      
                if sampled_token == end_token: 
                      break                
                decoded_caption = decoded_caption + f' {sampled_token}'

            text = re.sub(r'##', '', decoded_caption)
            text = re.sub(r'\s+', ' ', text)           
            print(f'Predicted report for image: {os.path.basename(pt)}', text)          

        return reports
   

