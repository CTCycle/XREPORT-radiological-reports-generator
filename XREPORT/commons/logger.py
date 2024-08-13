import os
import logging
import logging.config
from datetime import datetime

from XREPORT.commons.constants import LOGS_PATH

# Generate timestamp for the log filename
current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(LOGS_PATH, f'XREPORT_{current_timestamp}.log')

# Define logger configuration
###############################################################################
LOG_CONFIG =  {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%d-%m-%Y %H:%M:%S'
        },        
        'minimal': {
            'format': '%(levelname)s - %(message)s',
        },
    },    
    
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'minimal'            
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': log_filename,
            'mode': 'a'  
        },
    },
    'loggers': {
        'matplotlib': {
            'level': 'WARNING',
            'handlers': ['console', 'file'],
            'propagate': False
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file'],
    },    
}


# override logger configuration and load the latter
###############################################################################
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger()


