import os
import numpy as np
import tensorflow as tf 
from tqdm import tqdm   

from XREPORT.commons.utils.dataloader.serializer import DataSerializer, get_images_path
from XREPORT.commons.utils.preprocessing.tokenizers import BERTokenizer
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger



# [TOOLKIT TO USE THE PRETRAINED MODEL]
#------------------------------------------------------------------------------
class TextGenerator:

    def __init__(self, model):        

        np.random.seed(CONFIG["SEED"])
        tf.random.set_seed(CONFIG["SEED"])
        self.img_paths = get_images_path()
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]
        self.max_report_size = CONFIG["dataset"]["MAX_REPORT_SIZE"] + 1        
        self.dataserializer = DataSerializer()      
        self.model = model  

        self.layer_names = [layer.name for layer in model.layers]     
        self.encoder_layer_names = [x for x in self.layer_names if 'tranformer_encoder' in x] 
        self.decoder_layer_names = [x for x in self.layer_names if 'tranformer_decoder' in x] 

        # get tokenizers and its info
        tokenization = BERTokenizer()    
        self.tokenizer = tokenization.tokenizer 
    
    #--------------------------------------------------------------------------    
    def greed_search_generator(self):
        
        reports = {}
        vocabulary = self.tokenizer.get_vocab()
        start_token = '[CLS]'
        end_token = '[SEP]'

        # Convert start and end tokens to their corresponding indices
        start_token_idx = vocabulary[start_token]        
        index_lookup = {v: k for k, v in vocabulary.items()}  # Reverse the vocabulary mapping

        for pt in self.img_paths:
            logger.info(f'Generating report for image {os.path.basename(pt)}')
            image = tf.io.read_file(pt)
            image = tf.image.decode_image(image, channels=1)
            image = tf.image.resize(image, self.img_shape[:-1])
            image = image/255.0
            input_image = tf.expand_dims(image, 0)
            
            seq_input = np.zeros((1, self.max_report_size), dtype=np.int32)
            seq_input[0, 0] = start_token_idx  

            for i in tqdm(range(1, self.max_report_size)):                
                predictions = self.model.predict([input_image, seq_input], verbose=0)                
                next_token_idx = np.argmax(predictions[0, i-1, :], axis=-1)
                next_token = index_lookup[next_token_idx]
                
                # Stop if end token is generated
                if next_token == end_token:
                    break
                
                seq_input[0, i] = next_token_idx

            # Convert indices to tokens
            token_sequence = [index_lookup[idx] for idx in seq_input[0] if idx in index_lookup and idx != 0]
            reports[pt] = token_sequence
            logger.info(token_sequence)
           
        return reports