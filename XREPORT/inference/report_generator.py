
# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.utils.dataloader.serializer import ModelSerializer
from XREPORT.commons.utils.models.inferencer import TextGenerator


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD MODEL]
    #--------------------------------------------------------------------------     
    
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, parameters = modelserializer.load_pretrained_model()
    model_folder = modelserializer.loaded_model_folder
    model.summary(expand_nested=True)      
 
    # 2. [GENERATE REPORTS]
    #--------------------------------------------------------------------------
    print('Generate radiological reports for XRAY images\n') 
    generator = TextGenerator(model) 
    generated_reports = generator.greed_search_generator()




