import os
import numpy as np
import cv2
import matplotlib.pyplot as plt    


# [VALIDATION OF DATA]
#------------------------------------------------------------------------------
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
            

# [VALIDATION OF PRETRAINED MODELS]
#------------------------------------------------------------------------------
class ModelValidation:


    # model weights check
    #-------------------------------------------------------------------------- 
    def model_weigths_check(self, unsaved_model, save_path):

        # read model serialization configuration and initialize it           
        path = os.path.join(save_path, 'model', 'model_configuration.json')
        with open(path, 'r') as f:
            configuration = json.load(f)        
        saved_model = XREPCaptioningModel.from_config(configuration)             

        # set inputs to build the model 
        pic_shape = tuple(configuration['picture_shape'])
        sequence_length = configuration['sequence_length']
        build_inputs = (tf.constant(0.0, shape=(1, *pic_shape)),
                        tf.constant(0, shape=(1, sequence_length), dtype=tf.int32))
        saved_model(build_inputs, training=False) 

        # load weights into the model 
        weights_path = os.path.join(save_path, 'model', 'model_weights.h5')
        saved_model.load_weights(weights_path)        

        if len(unsaved_model.layers) != len(saved_model.layers):
            print('Models do not have the same number of layers')
            return

        # Iterate through each layer
        for layer1, layer2 in zip(unsaved_model.layers, saved_model.layers):
        # Check if both layers have weights
            if layer1.get_weights() and layer2.get_weights():
                # Compare weights
                weights1 = layer1.get_weights()[0] 
                weights2 = layer2.get_weights()[0] 
                biases1 = layer1.get_weights()[1] if len(layer1.get_weights()) > 1 else None
                biases2 = layer2.get_weights()[1] if len(layer2.get_weights()) > 1 else None
                
                # Using numpy to check if weights and biases are identical
                if not (np.array_equal(weights1, weights2) and (biases1 is None or np.array_equal(biases1, biases2))):
                    print(f'Weights or biases do not match in layer {layer1.name}')
                else:
                    print(f'Layer {layer1.name} weights and biases are identical')
            else:
                print(f'Layer {layer1.name} does not have weights to compare')
            

    
    


   


    