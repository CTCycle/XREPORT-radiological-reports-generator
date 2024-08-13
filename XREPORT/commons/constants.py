import json
from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'dataset')
IMG_DATA_PATH = join(RSC_PATH, 'dataset', 'images')
RESULTS_PATH = join(RSC_PATH, 'results')
TOKENIZER_PATH = join(RSC_PATH, 'tokenizer', 'BERT')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
GENERATION_INPUT_PATH = join(RSC_PATH, 'generation', 'input_images')
GENERATION_OUTPUT_PATH = join(RSC_PATH, 'generation', 'reports')
LOGS_PATH = join(PROJECT_DIR, 'resources', 'logs')
DATASET_NAME = 'XREPORT_dataset.csv'

CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)