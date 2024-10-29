import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from XREPORT.commons.utils.dataloader.serializer import DataSerializer
from XREPORT.commons.constants import CONFIG, VALIDATION_PATH
from XREPORT.commons.logger import logger


# [VALIDATION OF DATA]
###############################################################################
class ImageDatasetValidation:

    def __init__(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):
        self.DPI = 400
        self.file_type = 'jpg'
        self.train_data = train_data['path'].to_list()
        self.validation_data = validation_data['path'].to_list()
        self.serializer = DataSerializer()             
       
    #--------------------------------------------------------------------------
    def get_images_for_validation(self):

        train_images = (self.serializer.load_image(pt) 
                        for pt in self.train_data)
        validation_images = (self.serializer.load_image(pt) 
                             for pt in self.validation_data)

        return {'train' : train_images, 'validation' : validation_images}

    #--------------------------------------------------------------------------
    def pixel_intensity_histograms(self):

        images = self.get_images_for_validation()
        figure_path = os.path.join(VALIDATION_PATH, 'pixel_intensity_histograms.jpeg')
        plt.figure(figsize=(16, 14))        
        for name, image_set in images.items():
            pixel_intensities = np.concatenate([image.flatten() for image in image_set])
            plt.hist(pixel_intensities, bins='auto', alpha=0.5, label=name)        
        plt.title('Pixel Intensity Histograms', fontsize=16)
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()        
        plt.savefig(figure_path, bbox_inches='tight', 
                    format=self.file_type, dpi=self.DPI)
        plt.show()
        plt.close()

    #--------------------------------------------------------------------------
    def calculate_PSNR(self, img_path_1, img_path_2):
        
        img1 = cv2.imread(img_path_1)
        img2 = cv2.imread(img_path_2)       
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Calculate MSE
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:            
            return float('inf')      
               
        # Calculate PSNR
        PIXEL_MAX = 255.0 
        psnr = 20 * np.log10(PIXEL_MAX/np.sqrt(mse))

        return psnr
    
    
