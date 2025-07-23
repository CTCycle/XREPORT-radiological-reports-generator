import os
from dotenv import load_dotenv

from XREPORT.app.constants import PROJECT_DIR
from XREPORT.app.logger import logger

# [IMPORT CUSTOM MODULES]
###############################################################################
class EnvironmentVariables:

    def __init__(self):        
        self.env_path = os.path.join(PROJECT_DIR, 'setup', '.env')        
        if os.path.exists(self.env_path):
            load_dotenv(dotenv_path=self.env_path, override=True)            
        else:
            logger.error(f".env file not found at: {self.env_path}")   
    
    #--------------------------------------------------------------------------
    def get_environment_variables(self):                  
        return {"KERAS_BACKEND": os.getenv("KERAS_BACKEND", "torch"),
                "TF_CPP_MIN_LOG_LEVEL": os.getenv("TF_CPP_MIN_LOG_LEVEL", "1")}
       
