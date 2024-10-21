import os
import numpy as np
import torch
import keras
import tensorflow as tf
from tqdm import tqdm   

from XREPORT.commons.utils.dataloader.serializer import DataSerializer, get_images_path
from XREPORT.commons.utils.process.tokenizers import TokenWizard
from XREPORT.commons.constants import CONFIG, GENERATION_INPUT_PATH
from XREPORT.commons.logger import logger



# [TOOLKIT TO USE THE PRETRAINED MODEL]
###############################################################################
class TextGenerator:

    def __init__(self, model : keras.Model, configuration):        

        np.random.seed(configuration["SEED"])
        torch.manual_seed(configuration["SEED"])
        tf.random.set_seed(configuration["SEED"])
        self.img_paths = get_images_path(GENERATION_INPUT_PATH)
        self.img_shape = configuration["model"]["IMG_SHAPE"]
        self.max_report_size = configuration["dataset"]["MAX_REPORT_SIZE"]    
        self.dataserializer = DataSerializer()      
        self.model = model 
        self.configuration = configuration 

        self.layer_names = [layer.name for layer in model.layers]     
        self.encoder_layer_names = [x for x in self.layer_names if 'tranformer_encoder' in x] 
        self.decoder_layer_names = [x for x in self.layer_names if 'tranformer_decoder' in x] 

        # get tokenizers and its info
        tokenization = TokenWizard(configuration)    
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
            image = self.dataserializer.load_image(pt, self.img_shape)
            
            seq_input = keras.ops.zeros((1, self.max_report_size), dtype=torch.int32)
            seq_input[0, 0] = start_token_idx  

            for i in tqdm(range(1, self.max_report_size)):                
                predictions = self.model.predict([image, seq_input[:, :-1]], verbose=0)                
                next_token_idx = keras.ops.argmax(predictions[0, i-1, :], axis=-1).item()
                next_token = index_lookup[next_token_idx]                
                # Stop if end token is generated
                if next_token == end_token:
                    break
                
                seq_input[0, i] = next_token_idx

            # Convert indices to tokens
            token_sequence = [index_lookup[idx.item()] for idx in seq_input[0, :] if idx.item() in index_lookup and idx != 0]
            reports[pt] = token_sequence
            logger.info(token_sequence)
           
        return reports