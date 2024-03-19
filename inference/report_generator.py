import os
import sys

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and components
#------------------------------------------------------------------------------
from utils.data_assets import PreProcessing
from utils.model_assets import Inference
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
rep_path = os.path.join(globpt.inference_path, 'reports') 
cp_path = os.path.join(globpt.train_path, 'checkpoints') 
bert_path = os.path.join(globpt.train_path, 'BERT')
os.mkdir(rep_path) if not os.path.exists(rep_path) else None
os.mkdir(cp_path) if not os.path.exists(cp_path) else None 
os.mkdir(bert_path) if not os.path.exists(bert_path) else None 

# [LOAD MODEL AND DATA]
#==============================================================================
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
XREPORT report generation
-------------------------------------------------------------------------------
...
''')

preprocessor = PreProcessing()

# check report folder and generate list of images paths
#------------------------------------------------------------------------------
if not os.listdir(rep_path):
    print('No XRAY scans found in the report generation folder, please add them before continuing\n')
    sys.exit()
else:
    scan_paths = [os.path.join(root, file) for root, dirs, files in os.walk(rep_path) for file in files]
    print(f'XRAY images found: {len(scan_paths)}\n')

# Load pretrained model and tokenizer
#------------------------------------------------------------------------------
inference = Inference(cnf.seed) 
model, parameters = inference.load_pretrained_model(cp_path)
model_path = inference.folder_path
model.summary()

# load BioBERT tokenizer
tokenizer = preprocessor.get_BERT_tokenizer(bert_path)

# [GENERATE REPORTS]
#==============================================================================
#==============================================================================

# generate captions
#------------------------------------------------------------------------------ 
print('Generate the reports for XRAY images\n')
scan_size = tuple(parameters['picture_shape'][:-1])
vocab_size = parameters['vocab_size']
report_length = parameters['sequence_length']
generated_reports = inference.greed_search_generator(model, scan_paths, scan_size, 
                                                     report_length, tokenizer)


