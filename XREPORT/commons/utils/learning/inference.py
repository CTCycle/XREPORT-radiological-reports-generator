import os
import numpy as np
import torch
import keras
import tensorflow as tf
from tqdm import tqdm   

from XREPORT.commons.utils.dataloader.serializer import DataSerializer
from XREPORT.commons.utils.process.tokenizers import TokenWizard
from XREPORT.commons.constants import CONFIG, GENERATION_INPUT_PATH
from XREPORT.commons.logger import logger


# [TOOLKIT TO USE THE PRETRAINED MODEL]
###############################################################################
class TextGenerator:

    def __init__(self, model : keras.Model, configuration):          
       
        self.img_shape = configuration["model"]["IMG_SHAPE"]
        self.max_report_size = configuration["dataset"]["MAX_REPORT_SIZE"]
            
        self.dataserializer = DataSerializer(configuration)      
        self.model = model 
        self.configuration = configuration 

        self.layer_names = [layer.name for layer in model.layers]     
        self.encoder_layer_names = [x for x in self.layer_names if 'transformer_encoder' in x] 
        self.decoder_layer_names = [x for x in self.layer_names if 'transformer_decoder' in x] 

        # Get tokenizer and its info
        self.tokenization = TokenWizard(self.configuration)
        self.tokenizer = self.tokenization.tokenizer

        self.selected_method = configuration["inference"]["GEN_METHOD"]
        self.generator_methods = {'greedy' : self.greed_search_generator,
                                  'beam' : None}

        
    #--------------------------------------------------------------------------
    def get_tokenizer_parameters(self): 

        tokenizer_parameters = {"vocabulary_size": self.tokenization.vocabulary_size,
                                "start_token": self.tokenizer.cls_token,
                                "end_token": self.tokenizer.sep_token,
                                "pad_token": self.tokenizer.pad_token_id,
                                "start_token_idx": self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
                                "end_token_idx": self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)}
        
        return tokenizer_parameters
    
    #--------------------------------------------------------------------------    
    def merge_tokens(self, tokens : list[str]):

        processed_tokens = []
        for token in tokens:
            if token.startswith("##"):                
                processed_tokens[-1] += token[2:]
            else:                
                processed_tokens.append(token)
        
        joint_text = ' '.join(processed_tokens)
        
        return joint_text
    
    #--------------------------------------------------------------------------    
    def translate_tokens_to_text(self, index_lookup, sequence, tokenizer_config):

        # convert indexes to token using tokenizer vocabulary
        # define special tokens and remove them from generated tokens list
        token_sequence = [index_lookup[idx.item()] for idx in sequence[0, :] 
                          if idx.item() in index_lookup and idx != 0]
        
        special_tokens = [tokenizer_config["start_token"], 
                          tokenizer_config["end_token"],
                          tokenizer_config["pad_token"]]                          
       
        text_tokens = [token for token in token_sequence if token not in special_tokens]
        processed_text = self.merge_tokens(text_tokens)        
        
        return processed_text
    
    #--------------------------------------------------------------------------    
    def greed_search_generator(self, tokenizer_config, image_path):
        
        reports = {}
        vocabulary = self.tokenizer.get_vocab()
        start_token = tokenizer_config["start_token"]
        end_token = tokenizer_config["end_token"]
        pad_token = tokenizer_config["pad_token"]

        # Convert start and end tokens to their corresponding indices
        start_token_idx = tokenizer_config["start_token_idx"]        
        index_lookup = {v: k for k, v in vocabulary.items()}  
               
        logger.info(f'Generating report for image {os.path.basename(image_path)}')
        image = self.dataserializer.load_image(image_path)
        image = keras.ops.expand_dims(image, axis=0)
        
        seq_input = keras.ops.zeros((1, self.max_report_size), dtype=torch.int32)
        seq_input[0, 0] = start_token_idx  

        progress_bar = tqdm(total=self.max_report_size - 1)
        for i in range(1, self.max_report_size):         
            predictions = self.model.predict([image, seq_input[:, :-1]], verbose=0)                
            next_token_idx = keras.ops.argmax(predictions[0, i-1, :], axis=-1).item()
            next_token = index_lookup[next_token_idx]                
            # Stop if end token is generated
            if next_token == end_token:                
                
                progress_bar.n = progress_bar.total  # Set current progress to total
                progress_bar.last_print_n = progress_bar.total  # Update visual display
                progress_bar.update(0)  # Force update
                break
            
            seq_input[0, i] = next_token_idx
            progress_bar.update(1)
            
        progress_bar.close()
        report = self.translate_tokens_to_text(index_lookup, seq_input, tokenizer_config)                
        logger.info(report)
           
        return report    

    #--------------------------------------------------------------------------    
    def generate_radiological_reports(self, images_path):

        reports = {}         
        tokenizer_config = self.get_tokenizer_parameters()
        for path in images_path:
            report = self.generator_methods[self.selected_method](tokenizer_config, path)            
            reports[path] = report                

        return reports