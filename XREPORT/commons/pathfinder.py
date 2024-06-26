from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
DATA_PATH = join(PROJECT_DIR, 'data')
IMG_DATA_PATH = join(DATA_PATH, 'images')
VAL_PATH = join(DATA_PATH, 'validation')
TABLES_PATH = join(DATA_PATH, 'datasets')
CHECKPOINT_PATH = join(PROJECT_DIR, 'training', 'checkpoints')
BERT_PATH = join(PROJECT_DIR, 'training', 'BERT')
REPORT_PATH = join(PROJECT_DIR, 'inference', 'reports')

