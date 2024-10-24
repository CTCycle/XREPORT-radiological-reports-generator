# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.dataloader.serializer import ModelSerializer
from XREPORT.commons.utils.learning.inferencer import TextGenerator
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------  
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, configuration, history = modelserializer.load_pretrained_model()
    model_folder = modelserializer.loaded_model_folder
    model.summary(expand_nested=True)      
 
    # 2. [GENERATE REPORTS]
    #--------------------------------------------------------------------------    
    generator = TextGenerator(model, configuration) 
    generated_reports = generator.greed_search_generator()




