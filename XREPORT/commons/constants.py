import json
from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = abspath(join(__file__, "../.."))
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'database')
IMG_PATH = join(DATA_PATH, 'images')
METADATA_PATH = join(DATA_PATH, 'metadata')
VALIDATION_PATH = join(DATA_PATH, 'validation')
MODELS_PATH = join(RSC_PATH, 'models')
TOKENIZERS_PATH = join(MODELS_PATH, 'tokenizers')
ENCODERS_PATH = join(MODELS_PATH, 'XRAYEncoder')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
GENERATION_INPUT_PATH = join(RSC_PATH, 'inference', 'radiography')
GENERATION_OUTPUT_PATH = join(RSC_PATH, 'inference', 'reports')
LOGS_PATH = join(RSC_PATH, 'logs')

# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)

