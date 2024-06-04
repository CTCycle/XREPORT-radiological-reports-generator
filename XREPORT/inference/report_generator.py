import os
import sys

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.utils.preprocessing import PreProcessing
from XREPORT.utils.models import Inference
from XREPORT.config.pathfinder import REPORT_PATH, CHECKPOINT_PATH, BERT_PATH
import XREPORT.config.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------     
    preprocessor = PreProcessing()

    # check report folder and generate list of images paths    
    if not os.listdir(REPORT_PATH):
        print('No XRAY scans found in the report generation folder, please add them before continuing\n')
        sys.exit()
    else:
        scan_paths = [os.path.join(root, file) for root, dirs, files in os.walk(REPORT_PATH) for file in files]
        print(f'XRAY images found: {len(scan_paths)}\n')

    # Load pretrained model and tokenizer
    inference = Inference(cnf.SEED) 
    model, parameters = inference.load_pretrained_model(CHECKPOINT_PATH)
    model_path = inference.folder_path
    model.summary()

    # load BERT tokenizer
    tokenizer = preprocessor.get_BERT_tokenizer(BERT_PATH)
 
    # 2. [GENERATE REPORTS]
    #-------------------------------------------------------------------------- 
    # generate captions    
    print('Generate the reports for XRAY images\n')
    scan_size = tuple(parameters['picture_shape'][:-1])
    vocab_size = parameters['vocab_size']
    report_length = parameters['sequence_length']
    generated_reports = inference.greed_search_generator(model, scan_paths, scan_size, 
                                                         report_length, tokenizer)


