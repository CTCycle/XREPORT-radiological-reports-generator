import os
import numpy as np
import cv2
import matplotlib.pyplot as plt    


# [VALIDATION OF DATA]
#==============================================================================
# Series of methods and functions to preprocess data for model training
#==============================================================================
class DataValidation:

    #--------------------------------------------------------------------------
    def pixel_intensity_histograms(self, image_set_1, image_set_2, path, params,
                                   names=['First set', 'Second set']):
        
        '''
        Generates and saves histograms of pixel intensities for two sets of images.
        This function computes the pixel intensity distributions for two sets 
        of images and plots their histograms for comparison. The histograms are 
        saved as a JPEG file in the specified path. 

        Keyword Arguments:
            image_set_1 (list of ndarray): The first set of images for histogram comparison
            image_set_2 (list of ndarray): The second set of images for histogram comparison
            path (str): Directory path where the histogram image will be saved
            names (list of str, optional): Labels for the two image sets. Default to ['First set', 'Second set']
        
        Returns:
            None

        '''       
        pixel_intensities_1 = np.concatenate([image.flatten() for image in image_set_1])
        pixel_intensities_2 = np.concatenate([image.flatten() for image in image_set_2])        
        plt.hist(pixel_intensities_1, bins='auto', alpha=0.5, color='blue', label=names[0])
        plt.hist(pixel_intensities_2, bins='auto', alpha=0.5, color='red', label=names[1])
        plt.title(params['title'],)
        plt.xlabel('Pixel Intensity', fontsize=params['fontsize_labels'])
        plt.ylabel(params['ylabel'],  fontsize=params['fontsize_labels'])
        plt.legend()            
        plt.tight_layout()
        plot_loc = os.path.join(path, params['filename'])
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=400)   
        plt.show()         
        plt.close()

    #--------------------------------------------------------------------------
    def calculate_psnr(self, img_path_1, img_path_2):
        
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
            



   


    