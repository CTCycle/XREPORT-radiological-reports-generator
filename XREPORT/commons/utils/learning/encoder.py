from transformers import AutoImageProcessor, TFResNetModel
import torchvision.models as models

from XREPORT.commons.constants import CONFIG, ENCODERS_PATH
from XREPORT.commons.logger import logger


# [PRETRAINED IMAGE ENCODER]
###############################################################################
class ImageEncoder:

    def __init__(self):
        self.encoder_name = 'microsoft/resnet-50'

    # build model given the architecture
    #--------------------------------------------------------------------------
    def build_image_encoder(self):
        processor = AutoImageProcessor.from_pretrained(self.encoder_name, cache_dir=ENCODERS_PATH)
        model = TFResNetModel.from_pretrained(self.encoder_name, cache_dir=ENCODERS_PATH)

        inputs = processor(image, return_tensors="tf")      
        
        
        return processor, model   


