import os

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from XREPORT.app.src.interface.workers import check_thread_status, update_progress_callback
from XREPORT.app.src.constants import EVALUATION_PATH
from XREPORT.app.src.logger import logger


# [VALIDATION OF DATA]
###############################################################################
class TextAnalysis:

    def __init__(self, database, configuration : dict):
        self.DPI = 400
        self.file_type = 'jpg'
        self.database = database
        self.configuration = configuration  

    #--------------------------------------------------------------------------
    def count_words_in_documents(self, data):         
        words = [word for text in data['text'].to_list() for word in text.split()]           

        return words
    
    #--------------------------------------------------------------------------
    def calculate_text_statistics(self, data, **kwargs):
        images_descriptions = data['text'].to_list()
        images_path = data['path'].to_list()         
        results= []     
        for i, desc in enumerate(tqdm(
            images_descriptions, desc="Processing report", total=len(images_descriptions), ncols=100)):              
            results.append({'name': os.path.basename(images_path[i]),
                            'words_count': len(desc.split())})

            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, len(images_descriptions), kwargs.get('progress_callback', None))  

        stats_dataframe = pd.DataFrame(results) 
        self.database.save_text_statistics_table(stats_dataframe)       
        
        return stats_dataframe
    


        


# [VALIDATION OF DATA]
###############################################################################
class ImageAnalysis:

    def __init__(self, database, configuration):       
        self.database = database      
        self.save_images = configuration.get('save_images', True)          
        self.configuration = configuration      
        self.DPI = 400  

    #--------------------------------------------------------------------------
    def save_image(self, fig, name):        
        out_path = os.path.join(EVALUATION_PATH, name)
        fig.savefig(out_path, bbox_inches='tight', dpi=self.DPI)      
   
    #--------------------------------------------------------------------------
    def calculate_image_statistics(self, data, **kwargs):
        images_path = data['path'].to_list()         
        results= []     
        for i, path in enumerate(tqdm(
            images_path, desc="Processing images", total=len(images_path), ncols=100)):                  
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

            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, len(images_path), kwargs.get('progress_callback', None))  

        stats_dataframe = pd.DataFrame(results) 
        self.database.save_image_statistics_table(stats_dataframe)       
        
        return stats_dataframe
    
    #--------------------------------------------------------------------------
    def calculate_pixel_intensity_distribution(self, data, **kwargs):
        images_path = data['path'].to_list()               
        image_histograms = np.zeros(256, dtype=np.int64)        
        for i, path in enumerate(
            tqdm(images_path, desc="Processing image histograms", 
            total=len(images_path), ncols=100)):                        
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Calculate histogram for grayscale values [0, 255]
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            image_histograms += hist.astype(np.int64)
            
            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, len(images_path), kwargs.get('progress_callback', None))  

        # Plot the combined pixel intensity histogram
        fig, ax = plt.subplots(figsize=(16,14), dpi=self.DPI)
        plt.bar(np.arange(256),image_histograms, alpha=0.7)
        ax.set_title('Combined Pixel Intensity Histogram', fontsize=24)
        ax.set_xlabel('Pixel Intensity', fontsize=16, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')        
        ax.tick_params(axis='both', which='major', labelsize=14, labelcolor='black')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.tight_layout()        
        self.save_image(fig, "pixels_intensity_histogram.jpeg") 
        plt.close()          

        return fig     

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
    


    
    
