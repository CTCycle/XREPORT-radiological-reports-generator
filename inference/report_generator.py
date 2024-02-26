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
from components.data_assets import PreProcessing
from components.model_assets import Inference
import components.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
rep_path = os.path.join(globpt.inference_path, 'reports') 
cp_path = os.path.join(globpt.train_path, 'checkpoints') 
os.mkdir(rep_path) if not os.path.exists(rep_path) else None
os.mkdir(cp_path) if not os.path.exists(cp_path) else None  

# [LOAD MODEL AND DATA]
#==============================================================================
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
XREPORT report generation
-------------------------------------------------------------------------------
...
''')

# check report folder and generate list of images paths
#------------------------------------------------------------------------------
if not os.listdir(rep_path):
    print('''No XRAY scans found in the report generation folder, please add them before continuing,
the script will now be closed!\n''')
    sys.exit()
else:
    scan_paths = [os.path.join(root, file) for root, dirs, files in os.walk(rep_path) for file in files]
    print(f'''XRAY images found: {len(scan_paths)}
Report generation will start once you've selected the model.\n''')    

# Load pretrained model and its parameters
#------------------------------------------------------------------------------
inference = Inference(cnf.seed) 
model, parameters = inference.load_pretrained_model(cp_path)
model_path = inference.folder_path
model.summary()

# Load the tokenizer
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
pp_path = os.path.join(model_path, 'preprocessing')
tokenizer = preprocessor.load_tokenizer(pp_path, 'word_tokenizer')

# [GENERATE REPORTS]
#==============================================================================
#==============================================================================
print('''Generate the reports for XRAY images
''')

# generate captions
#------------------------------------------------------------------------------ 
scan_size = tuple(parameters['picture_shape'][:-1])
channels = parameters['picture_shape'][-1]
vocab_size = parameters['vocab_size']
report_length = parameters['sequence_length']
generated_reports = inference.generate_reports(model, scan_paths, channels, scan_size, report_length, tokenizer)


