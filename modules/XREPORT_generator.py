import os
import sys

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path if this file is launched as __main__
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import modules and components
#------------------------------------------------------------------------------
from modules.components.data_assets import PreProcessing
from modules.components.training_assets import InferenceTools
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD MODEL AND DATA]
#==============================================================================
# ....
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
XREPORT generation
-------------------------------------------------------------------------------
...
''')

# check report folder and generate list of images paths
#------------------------------------------------------------------------------
if not os.listdir(GlobVar.rep_path):
    print('''No XRAY scans found in the report generation folder, please add them before continuing,
the script will now be closed!
''')
    sys.exit()
else:
    scan_paths = [os.path.join(root, file) for root, dirs, files in os.walk(GlobVar.rep_path) for file in files]
    print(f'''XRAY images found: {len(scan_paths)}
Report generation will start once you've selected the model.
''')    

# Load pretrained model and its parameters
#------------------------------------------------------------------------------
inference = InferenceTools() 
model, configurations = inference.load_pretrained_model(GlobVar.model_path)
model.summary()

# Load the tokenizer
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
tokenizer_path = os.path.join(GlobVar.data_path, 'Tokenizers')
tokenizer = preprocessor.load_tokenizer(tokenizer_path, 'word_tokenizer')

# [GENERATE REPORTS]
#==============================================================================
# ....
#==============================================================================
print('''Generate the reports for XRAY images
''')

# generate captions
#------------------------------------------------------------------------------ 
scan_size = tuple(configurations['pic_shape'][:-1])
channels = configurations['pic_shape'][-1]
vocab_size = configurations['vocab_size']
report_length = configurations['sequence_length']
generated_reports = inference.generate_reports(model, scan_paths, channels, scan_size, report_length, tokenizer)


