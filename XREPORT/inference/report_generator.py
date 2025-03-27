# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.inference.generator import TextGenerator
from XREPORT.commons.constants import CONFIG, GENERATION_INPUT_PATH
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------  
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()    
    model.summary(expand_nested=True)          
 
    # 2. [GENERATE REPORTS]
    # One can select different either greedy_search or beam search to genarate
    # reports with a pretrained decoder 
    #--------------------------------------------------------------------------
    dataserializer = DataSerializer(CONFIG)  
    img_paths = dataserializer.get_images_path_from_folder(GENERATION_INPUT_PATH)
    logger.info(f'\nStart generating reports using model {os.path.basename(checkpoint_path)}')
    logger.info(f'{len(img_paths)} images have been found and are ready for inference pipeline')    
    generator = TextGenerator(model, configuration) 
    generated_reports = generator.generate_radiological_reports(img_paths)




