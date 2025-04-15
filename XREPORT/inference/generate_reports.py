# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.learning.training import ModelTraining
from XREPORT.commons.utils.inference.generator import TextGenerator
from XREPORT.commons.constants import CONFIG, INFERENCE_INPUT_PATH
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------  
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, configuration, metadata, _, checkpoint_path = modelserializer.select_and_load_checkpoint()    
    model.summary(expand_nested=True)  

    # setting device for training    
    trainer = ModelTraining(configuration)    
    trainer.set_device()        
     
    # 2. [GET IMAGES]
    #--------------------------------------------------------------------------   
    # select images from the inference folder and retrieve current paths
    dataserializer = DataSerializer(CONFIG)  
    img_paths = dataserializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
    logger.info(f'\nStart generating reports using model {os.path.basename(checkpoint_path)}')
    logger.info(f'{len(img_paths)} images have been found and are ready for inference pipeline') 

    # 3. [GENERATE REPORTS]
    # One can select different either greedy_search or beam search to genarate
    # reports with a pretrained decoder 
    #--------------------------------------------------------------------------
    # generate radiological reports from the list of inference image paths   
    generator = TextGenerator(model, configuration, checkpoint_path) 
    generated_reports = generator.generate_radiological_reports(img_paths)
    dataserializer.save_generated_reports(generated_reports)




