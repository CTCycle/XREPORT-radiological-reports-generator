# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.dataloader.serializer import get_images_path, ModelSerializer
from XREPORT.commons.utils.learning.inference import TextGenerator
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
    print()   
 
    # 2. [GENERATE REPORTS]
    # One can select different either greedy_search or beam search to genarate
    # reports with a pretrained decoder 
    #--------------------------------------------------------------------------  
    img_paths = get_images_path(GENERATION_INPUT_PATH)
    logger.info(f'Start generating reports using model {os.path.basename(checkpoint_path)}')
    logger.info(f'{len(img_paths)} images have been found in resources/encoding/images')    
    generator = TextGenerator(model, configuration) 
    generated_reports = generator.generate_radiological_reports(img_paths)




