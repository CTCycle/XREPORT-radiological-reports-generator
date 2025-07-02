import os
import shutil
import pandas as pd

from nltk.translate.bleu_score import corpus_bleu

from XREPORT.commons.utils.learning.inference.generator import TextGenerator
from XREPORT.commons.utils.learning.callbacks import InterruptTraining
from XREPORT.commons.utils.data.serializer import ModelSerializer
from XREPORT.commons.interface.workers import check_thread_status, update_progress_callback
from XREPORT.commons.constants import CHECKPOINT_PATH
from XREPORT.commons.logger import logger


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, database, configuration, remove_invalid=False):
        self.remove_invalid = remove_invalid            
        self.database = database      
        self.configuration = configuration

    #---------------------------------------------------------------------------
    def scan_checkpoint_folder(self):
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():                
                pretrained_model_path = os.path.join(entry.path, 'saved_model.keras')                
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)
                elif not os.path.isfile(pretrained_model_path) and self.remove_invalid:                    
                    shutil.rmtree(entry.path)

        return model_paths  

    #---------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs):    
        serializer = ModelSerializer()            
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for i, model_path in enumerate(model_paths):                
            model = serializer.load_checkpoint(model_path)
            configuration, history = serializer.load_training_configuration(model_path)
            model_name = os.path.basename(model_path)                   
            precision = 16 if configuration.get("use_mixed_precision", 'NA') else 32 
            chkp_config = {'Sample size': configuration.get("train_sample_size", 'NA'),
                           'Validation size': configuration.get("validation_size", 'NA'),
                           'Seed': configuration.get("train_seed", 'NA'),                           
                           'Precision (bits)': precision,                      
                           'Epochs': configuration.get("epochs", 'NA'),
                           'Additional Epochs': configuration.get("additional_epochs", 'NA'),
                           'Batch size': configuration.get("batch_size", 'NA'),           
                           'Split seed': configuration.get("split_seed", 'NA'),
                           'Image augmentation': configuration.get("img_augmentation", 'NA'),
                           'Image height': 224,
                           'Image width': 224,
                           'Image channels': 3,                          
                           'JIT Compile': configuration.get("jit_compile", 'NA'),                           
                           'Device': configuration.get("device", 'NA'),                                                      
                           'Number workers': configuration.get("num_workers", 'NA'),
                           'LR Scheduler': configuration.get("use_scheduler", 'NA'),                            
                           'LR Scheduler - Post Warmup LR': configuration.get("post_warmup_LR", 'NA'),
                           'LR Scheduler - Warmup Steps': configuration.get("warmup_steps", 'NA'),
                           'Temperature': configuration.get("train_temperature", 'NA'),                            
                           'Tokenizer': configuration["dataset"].get("TOKENIZER", 'NA'),                            
                           'Max report size': configuration["dataset"].get("MAX_REPORT_SIZE", 'NA'),
                           'Number of heads': configuration["model"].get("ATTENTION_HEADS", 'NA'),
                           'Number of encoders': configuration["model"].get("NUM_ENCODERS", 'NA'),
                           'Number of decoders': configuration["model"].get("NUM_DECODERS", 'NA'),
                           'Embedding dimensions': configuration["model"].get("EMBEDDING_DIMS", 'NA'),
                           'Frozen image encoder': configuration["model"].get("FREEZE_IMG_ENCODER", 'NA')}

            model_parameters.append(chkp_config)

            # check for thread status and progress bar update   
            check_thread_status(kwargs.get('worker', None))         
            update_progress_callback(
                i, len(model_paths), kwargs.get('progress_callback', None)) 


        dataframe = pd.DataFrame(model_parameters)
        self.database.save_checkpoints_summary_table(dataframe)        
            
        return dataframe
    
    #--------------------------------------------------------------------------
    def get_evaluation_report(self, model, validation_dataset, **kwargs):
        callbacks_list = [InterruptTraining(kwargs.get('worker', None))]
        validation = model.evaluate(validation_dataset, verbose=1, callbacks=callbacks_list)      
        logger.info(
        f'Sparse Categorical Entropy Loss {validation[0]:.3f} - Sparse Categorical Accuracy {validation[1]:.3f}') 
    

    
# [VALIDATION OF DATA]
###############################################################################
class EvaluateTextConsistency:

    def __init__(self, model, configuration : dict):
        self.model = model
        self.configuration = configuration
        self.num_samples = 50
        self.generator = TextGenerator(model, configuration, verbose=False)
        
        self.tokenizer_config = self.generator.get_tokenizer_parameters()

    #--------------------------------------------------------------------------
    def calculate_BLEU_score(self, validation_data, **kwargs):
        samples = validation_data.sample(n=self.num_samples, random_state=42) 
        sampled_images = samples['path'].to_list()     
        true_reports = dict(zip(samples['path'], samples['text']))
        
        # Generate reports using greedy decoding
        generated_with_greedy = self.generator.generate_radiological_reports(
            sampled_images, override_method='greedy')        
        
        references = []
        hypotheses = []
        
        # For each image, tokenize the corresponding ground-truth and generated reports.
        for i, image in enumerate(sampled_images):
            # Ensure that the image key exists in both the true reports and generated dictionary.
            if image in generated_with_greedy and image in true_reports:
                # Tokenize using simple split (or use nltk.word_tokenize for better tokenization)
                ref_tokens = true_reports[image].lower().split()  # or use: nltk.word_tokenize(true_reports[image].lower())
                cand_tokens = generated_with_greedy[image].lower().split()  # or use: nltk.word_tokenize(generated_with_greedy[image].lower())
                references.append([ref_tokens])  # each reference is a list of tokens; nested in a list to support multiple refs
                hypotheses.append(cand_tokens)

            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, sampled_images, kwargs.get('progress_callback', None))  

        
        # Calculate corpus BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        logger.info(f'BLEU score for {self.num_samples} validation samples: {bleu_score}')

        return bleu_score