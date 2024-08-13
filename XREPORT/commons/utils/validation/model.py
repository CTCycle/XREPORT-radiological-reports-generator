import os
import cv2
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class ModelValidation:

    def __init__(self, model : keras.Model):
        self.DPI = 400
        self.file_type = 'jpg'        
        self.model = model

    #-------------------------------------------------------------------------- 
    def evaluation_report(self, train_dataset, validation_dataset):
        
        train_eval = self.model.evaluate(train_dataset, verbose=1)
        validation_eval = self.model.evaluate(validation_dataset, verbose=1)
        logger.info('Train dataset:')
        logger.info(f'Loss: {train_eval[0]}')    
        logger.info(f'Metric: {train_eval[1]}')  
        logger.info('Test dataset:')
        logger.info(f'Loss: {validation_eval[0]}')    
        logger.info(f'Metric: {validation_eval[1]}') 

    #-------------------------------------------------------------------------- 
    def visualize_features_vector(self, real_image, features, predicted_image, name, path):
        
        fig_path = os.path.join(path, f'{name}.jpeg')
        fig, axs = plt.subplots(1, 3, figsize=(14, 20), dpi=600)                                     
        axs[0].imshow(real_image[0])
        axs[0].set_title('Original picture')
        axs[0].axis('off')
        axs[1].imshow(features)
        axs[1].set_title('Extracted features')
        axs[1].axis('off')
        axs[2].imshow(predicted_image[0])
        axs[2].set_title('Reconstructed picture')
        axs[2].axis('off')
        plt.tight_layout() 
        plt.show(block=False)       
        plt.savefig(fig_path, bbox_inches='tight', format=self.file_type, dpi=self.DPI)                
        plt.close()
        
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, dataset : tf.data.Dataset, name, path):

        # perform visual validation for the train dataset (initialize a validation tf.dataset
        # with batch size of 10 images)
        logger.info('Visual reconstruction evaluation: train dataset')        
        batch = dataset.take(1)
        for images, labels in batch:
            recostructed_images = self.model.predict(images, verbose=0)  
            eval_path = os.path.join(path, 'data')
            num_pics = len(images)
            fig_path = os.path.join(eval_path, f'{name}.jpeg')
            fig, axs = plt.subplots(num_pics, 2, figsize=(4, num_pics * 2))
            for i, (real, pred) in enumerate(zip(images, recostructed_images)):                                                          
                axs[i, 0].imshow(real)
                if i == 0:
                    axs[i, 0].set_title('Original picture')
                axs[i, 0].axis('off')
                axs[i, 1].imshow(pred)
                if i == 0:
                    axs[i, 1].set_title('Reconstructed picture')
                axs[i, 1].axis('off')
            plt.tight_layout() 
            plt.show(block=False)       
            plt.savefig(fig_path, bbox_inches='tight', format=self.file_type, dpi=self.DPI)               
            plt.close()
        

              
        
        
            
        
