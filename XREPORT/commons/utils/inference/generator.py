import os
import torch
import keras
import numpy as np
from tqdm import tqdm   

from XREPORT.commons.utils.data.loader import InferenceDataLoader
from XREPORT.commons.utils.data.process.tokenizers import TokenWizard
from XREPORT.commons.logger import logger


# [TOOLKIT TO USE THE PRETRAINED MODEL]
###############################################################################
class TextGenerator:

    def __init__(self, model : keras.Model, configuration, path=None, verbose=True):
        keras.utils.set_random_seed(configuration["SEED"])  
        self.model = model 
        self.configuration = configuration        
        self.dataloader = InferenceDataLoader(configuration)
        self.name = os.path.basename(path) if path is not None else None 
        self.verbose = verbose       
        
        # define image and text parameters for inference
        self.img_shape = (224, 224)
        self.num_channels = 3           
        self.max_report_size = configuration["dataset"]["MAX_REPORT_SIZE"]  
        # get encoder and decoder layers names from loaded model
        self.layer_names = [layer.name for layer in model.layers]     
        self.encoder_layer_names = [
            x for x in self.layer_names if 'transformer_encoder' in x] 
        self.decoder_layer_names = [
            x for x in self.layer_names if 'transformer_decoder' in x] 

        # Get tokenizer and related configuration
        self.tokenization = TokenWizard(self.configuration)
        self.tokenizer = self.tokenization.tokenizer
        # report generation methods 
        self.selected_method = configuration["inference"]["GEN_METHOD"]
        self.generator_methods = {'greedy' : self.greed_search_generator,
                                  'beam' : self.beam_search_generator}

        
    #--------------------------------------------------------------------------
    def get_tokenizer_parameters(self): 
        tokenizer_parameters = {"vocabulary_size": self.tokenization.vocabulary_size,
                                "start_token": self.tokenizer.cls_token,
                                "end_token": self.tokenizer.sep_token,
                                "pad_token": self.tokenizer.pad_token_id,
                                "start_token_idx": self.tokenizer.convert_tokens_to_ids(
                                    self.tokenizer.cls_token),
                                "end_token_idx": self.tokenizer.convert_tokens_to_ids(
                                    self.tokenizer.sep_token)}
        
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
        # extract vocabulary from the tokenizers
        vocabulary = self.tokenizer.get_vocab()
        start_token = tokenizer_config["start_token"]
        end_token = tokenizer_config["end_token"]
        pad_token = tokenizer_config["pad_token"]

        # Convert start and end tokens to their corresponding indices
        start_token_idx = tokenizer_config["start_token_idx"]        
        index_lookup = {v: k for k, v in vocabulary.items()}  
               
        logger.info(f'Generating report for image {os.path.basename(image_path)}')
        image = self.dataloader.load_image_as_array(image_path)
        image = keras.ops.expand_dims(image, axis=0)
        # initialize an array with same size of max expected report length
        # set the start token as the first element
        seq_input = keras.ops.zeros((1, self.max_report_size), dtype='int32')
        seq_input[0, 0] = start_token_idx  
        # initialize progress bar for better output formatting
        progress_bar = tqdm(total=self.max_report_size - 1)
        for i in range(1, self.max_report_size): 
            # predict the next token based on the truncated sequence (last token removed)         
            predictions = self.model.predict([image, seq_input[:, :-1]], verbose=0)  
            # apply argmax (greedy search) to identify the most probable token              
            next_token_idx = keras.ops.argmax(predictions[0, i-1, :], axis=-1).item()
            next_token = index_lookup[next_token_idx]                
            # Stop sequence generation if end token is generated
            if next_token == end_token:               
                progress_bar.n = progress_bar.total  # Set current progress to total
                progress_bar.last_print_n = progress_bar.total  # Update visual display
                progress_bar.update(0)  # Force update
                break
            
            seq_input[0, i] = next_token_idx
            progress_bar.update(1)
            
        progress_bar.close()
        report = self.translate_tokens_to_text(
            index_lookup, seq_input, tokenizer_config)     

        logger.info(report) if self.verbose else None
           
        return report 

    #--------------------------------------------------------------------------
    def beam_search_generator(self, tokenizer_config, image_path, beam_width=3):
        # Extract tokenizer and token parameters
        vocabulary = self.tokenizer.get_vocab()
        start_token = tokenizer_config["start_token"]
        end_token = tokenizer_config["end_token"]
        start_token_idx = tokenizer_config["start_token_idx"]
        end_token_idx = tokenizer_config["end_token_idx"]
        index_lookup = {v: k for k, v in vocabulary.items()}

        logger.info(f'Generating report for image {os.path.basename(image_path)}')
        image = self.dataloader.load_image_as_array(image_path)
        image = keras.ops.expand_dims(image, axis=0)

        # Initialize the beam with a single sequence containing only the start token and a score of 0.0 (log-prob)
        beams = [([start_token_idx], 0.0)]
        
        # Loop over the maximum report length
        for step in range(1, self.max_report_size):
            new_beams = []
            # Expand each beam in the current list
            for seq, score in beams:
                # If the sequence has already generated the end token, carry it forward unchanged.
                if seq[-1] == end_token_idx:
                    new_beams.append((seq, score))
                    continue

                # Prepare a padded sequence input.
                # We create an array of zeros with shape (1, max_report_size) and fill in the current sequence.
                seq_input = keras.ops.zeros((1, self.max_report_size), dtype='int32')
                for j, token in enumerate(seq):
                    seq_input[0, j] = token

                # Use only the part of the sequence that has been generated so far.
                # (Following your greedy method, the model expects a truncated sequence, excluding the final slot.)
                current_input = seq_input[:, :len(seq)]
                predictions = self.model.predict([image, current_input], verbose=0)
                # Get the prediction corresponding to the last token in the sequence.
                # In your greedy search, predictions[0, i-1, :] was used; here len(seq)-1 corresponds to the same position.
                next_token_logits = predictions[0, len(seq)-1, :]
                # Convert logits/probabilities to log probabilities.
                # We clip to avoid log(0) issues.
                log_probs = np.log(np.clip(next_token_logits, 1e-12, 1.0))
                # Select the top `beam_width` token indices.
                top_indices = np.argsort(log_probs)[-beam_width:][::-1]

                # Create new beams for each candidate token.
                for token_idx in top_indices:
                    new_seq = seq + [int(token_idx)]
                    new_score = score + log_probs[token_idx]
                    new_beams.append((new_seq, new_score))
                    
            # Sort all new beams by their cumulative score (in descending order) and keep the top `beam_width` beams.
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]            
            # If every beam in the list ends with the end token, we can stop early.
            if all(seq[-1] == end_token_idx for seq, _ in beams):
                break

        # Choose the best beam (the one with the highest score)
        best_seq, best_score = beams[0]        
        # Create a full padded sequence from the best beam for conversion to text.
        seq_input = keras.ops.zeros((1, self.max_report_size), dtype='int32')
        for i, token in enumerate(best_seq):
            seq_input[0, i] = token

        report = self.translate_tokens_to_text(index_lookup, seq_input, tokenizer_config)
        logger.info(report) if self.verbose else None

        return report   

    #--------------------------------------------------------------------------    
    def generate_radiological_reports(self, images_path, override_method=None):
        reports = {}         
        tokenizer_config = self.get_tokenizer_parameters()
        for path in images_path:
            selected_method = self.selected_method if override_method is None else override_method
            report = self.generator_methods[selected_method](tokenizer_config, path)            
            reports[path] = report                

        return reports