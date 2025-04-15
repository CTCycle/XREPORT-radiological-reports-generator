import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu

from XREPORT.commons.utils.inference.generator import TextGenerator
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [VALIDATION OF DATA]
###############################################################################
class TextAnalysis:

    def __init__(self):
        self.DPI = 400
        self.file_type = 'jpg'

    #--------------------------------------------------------------------------
    def count_words_in_documents(self, data : pd.DataFrame):         
        words = [word for text in data['text'].to_list() for word in text.split()]           

        return words
    

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
    def calculate_BLEU_score(self, validation_data : pd.DataFrame):
        samples = validation_data.sample(n=self.num_samples, random_state=42) 
        sampled_images = samples['path'].to_list()     
        true_reports = dict(zip(samples['path'], samples['text']))
        
        # Generate reports using greedy decoding
        generated_with_greedy = self.generator.generate_radiological_reports(
            sampled_images, override_method='greedy')        
        
        references = []
        hypotheses = []
        
        # For each image, tokenize the corresponding ground-truth and generated reports.
        for image in sampled_images:
            # Ensure that the image key exists in both the true reports and generated dictionary.
            if image in generated_with_greedy and image in true_reports:
                # Tokenize using simple split (or use nltk.word_tokenize for better tokenization)
                ref_tokens = true_reports[image].lower().split()  # or use: nltk.word_tokenize(true_reports[image].lower())
                cand_tokens = generated_with_greedy[image].lower().split()  # or use: nltk.word_tokenize(generated_with_greedy[image].lower())
                references.append([ref_tokens])  # each reference is a list of tokens; nested in a list to support multiple refs
                hypotheses.append(cand_tokens)
        
        # Calculate corpus BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        logger.info(f'BLEU score for {self.num_samples} validation samples: {bleu_score}')

        return bleu_score
        