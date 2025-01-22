import json
from os.path import join, dirname, abspath 

# [PATHS]
###############################################################################
PROJECT_DIR = dirname(dirname(abspath(__file__)))
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'dataset')
IMG_DATA_PATH = join(RSC_PATH, 'dataset', 'images')
VALIDATION_PATH = join(RSC_PATH, 'validation')
TOKENIZERS_PATH = join(RSC_PATH, 'tokenizers')
ENCODERS_PATH = join(RSC_PATH, 'encoders')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
GENERATION_INPUT_PATH = join(RSC_PATH, 'generation', 'radiography')
GENERATION_OUTPUT_PATH = join(RSC_PATH, 'generation', 'reports')
LOGS_PATH = join(PROJECT_DIR, 'resources', 'logs')

# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)

