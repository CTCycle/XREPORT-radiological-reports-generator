from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, 'XREPORT')
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'database')
SOURCE_PATH = join(DATA_PATH, 'dataset')
IMG_PATH = join(SOURCE_PATH, 'images')
METADATA_PATH = join(DATA_PATH, 'metadata')
EVALUATION_PATH = join(DATA_PATH, 'validation')
MODELS_PATH = join(RSC_PATH, 'models')
TOKENIZERS_PATH = join(MODELS_PATH, 'tokenizers')
ENCODERS_PATH = join(MODELS_PATH, 'XRAYEncoder')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
INFERENCE_INPUT_PATH = join(DATA_PATH, 'inference')
LOGS_PATH = join(RSC_PATH, 'logs')

# [UI LAYOUT PATH]
###############################################################################
UI_PATH = join(PROJECT_DIR, 'app', 'assets', 'window_layout.ui')
QSS_PATH = join(PROJECT_DIR, 'app', 'assets', 'stylesheet.qss')

