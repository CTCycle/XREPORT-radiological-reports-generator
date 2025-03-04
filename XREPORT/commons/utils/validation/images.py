import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from XREPORT.commons.utils.dataloader.serializer import DataSerializer
from XREPORT.commons.constants import CONFIG, VALIDATION_PATH
from XREPORT.commons.logger import logger


# [VALIDATION OF DATA]
###############################################################################
class ImageAnalysis:

    def __init__(self):               
        self.serializer = DataSerializer()
        self.statistics_path = os.path.join(VALIDATION_PATH, 'dataset')
        self.plot_path = os.path.join(self.statistics_path, 'figures')
        os.makedirs(self.statistics_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)
        self.DPI = 400
        self.file_type = 'jpg'        
       
    #--------------------------------------------------------------------------
    def get_images_for_validation(self, data : pd.DataFrame):
        images_paths = data['path'].to_list()
        images = (self.serializer.load_image(pt) for pt in images_paths)        

        return images

    #--------------------------------------------------------------------------
    def calculate_image_statistics(self, data : pd.DataFrame):
        image_paths = data['path'].to_list()


    #--------------------------------------------------------------------------
    def calculate_image_statistics(self, data : pd.DataFrame): 
        images_path = data['path'].to_list()         
        results= []     
        for path in tqdm(
            images_path, desc="Processing images", total=len(images_path), ncols=100):                  
            img = cv2.imread(path)
            
            if img is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Convert image to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Get image dimensions
            height, width = gray.shape
            # Compute basic statistics
            mean_val = np.mean(gray)
            median_val = np.median(gray)
            std_val = np.std(gray)
            min_val = np.min(gray)
            max_val = np.max(gray)
            pixel_range = max_val - min_val

            # Estimate noise by comparing the image to a blurred version
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            noise = gray.astype(np.float32) - blurred.astype(np.float32)
            noise_std = np.std(noise)
            # Define the noise ratio (avoiding division by zero with a small epsilon)
            noise_ratio = noise_std / (std_val + 1e-9)
          
            results.append({'name': os.path.basename(path),
                            'height': height,
                            'width': width,
                            'mean': mean_val,
                            'median': median_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'pixel_range': pixel_range,
                            'noise_std': noise_std,
                            'noise_ratio': noise_ratio})           
        
        stats_dataframe = pd.DataFrame(results)
        csv_path = os.path.join(self.statistics_path, 'image_statistics.csv')
        stats_dataframe.to_csv(csv_path, index=False, sep=';', encoding='utf-8')
        
        return results
    
    #--------------------------------------------------------------------------
    def calculate_pixel_intensity(self, data : pd.DataFrame):
        images_path = data['path'].to_list()               
        image_histograms = np.zeros(256, dtype=np.int64)        
        for path in tqdm(
            images_path, desc="Processing image histograms", total=len(images_path), ncols=100):            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Calculate histogram for grayscale values [0, 255]
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            image_histograms += hist.astype(np.int64)

        # Plot the combined histogram
        plt.figure(figsize=(14, 12))
        plt.bar(np.arange(256),image_histograms, alpha=0.7)
        plt.title('Combined Pixel Intensity Histogram', fontsize=16)
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plot_path, 'pixel_intensity_histogram.jpeg'), 
            dpi=self.DPI)
        plt.close()        

        return image_histograms    

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
    


    
    
