import cv2
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from XREPORT.commons.utils.data.serializer import DataSerializer
from XREPORT.commons.utils.validation.images import ImageAnalysis
from XREPORT.commons.utils.data.loader import TrainingDataLoader
from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.data.splitting import TrainValidationSplit
from XREPORT.commons.utils.learning.training import ModelTraining
from XREPORT.commons.utils.learning.autoencoder import XREPORTAutoEncoder
from XREPORT.commons.utils.inference.encoding import ImageEncoding
from XREPORT.commons.utils.validation.reports import log_training_report
from XREPORT.commons.constants import DATA_PATH, IMG_PATH, INFERENCE_INPUT_PATH
from XREPORT.commons.logger import logger



###############################################################################
class ValidationEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)   
        self.analyzer = ImageAnalysis(configuration)     
        self.configuration = configuration  

    #--------------------------------------------------------------------------
    def load_images_path(self):
        sample_size = self.configuration.get("sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(
            IMG_PATH, sample_size) 
        
        return images_paths 
        
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None):
        sample_size = self.configuration.get("sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        logger.info(f'The image dataset is composed of {len(images_paths)} images')
        
        images = []        
        if 'image_stats' in metrics:
            logger.info('Current metric: image dataset statistics')
            image_statistics = self.analyzer.calculate_image_statistics(
                images_paths, progress_callback=progress_callback)
             
        if 'pixels_distribution' in metrics:
            logger.info('Current metric: pixel intensity distribution')
            images.append(self.analyzer.calculate_pixel_intensity_distribution(
                images_paths, progress_callback=progress_callback))       

        return images     

    #--------------------------------------------------------------------------
    def convert_fig_to_qpixmap(self, fig):    
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        # get the size in pixels and initialize raw RGBA buffer
        width, height = canvas.get_width_height()        
        buf = canvas.buffer_rgba()
        # construct a QImage pointing at that memory (no PNG decoding)
        qimg = QImage(buf, width, height, QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)
    
    #--------------------------------------------------------------------------    
    def load_image_as_pixmap(self, path):    
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Handle grayscale, RGB, or RGBA
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if img.shape[2] == 3:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        else:  # RGBA
            qimg = QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   

###############################################################################
class TrainingEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)        
        self.modser = ModelSerializer()         
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        return self.modser.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None):  
        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = self.configuration.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)

        splitter = TrainValidationSplit(self.configuration) 
        train_data, validation_data = splitter.split_train_and_validation(images_paths)
        
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing')     
        builder = TrainingDataLoader(self.configuration)          
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, validation_data)
        
        # set device for training operations based on user configuration        
        logger.info('Setting device for training operations based on user configuration') 
        trainer = ModelTraining(self.configuration)           
        trainer.set_device()

        # build the autoencoder model 
        logger.info('Building XREPORT AutoEncoder model based on user configuration') 
        checkpoint_path = self.modser.create_checkpoint_folder()
        autoencoder = XREPORTAutoEncoder(self.configuration)           
        model = autoencoder.get_model(model_summary=True) 

        # generate training log report and graphviz plot for the model layout         
        log_training_report(train_data, validation_data, self.configuration)        
        self.modser.save_model_plot(model, checkpoint_path) 
        # perform training and save model at the end
        logger.info('Starting XREPORT AutoEncoder training') 
        trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, 
            progress_callback=progress_callback)
        
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')         
        model, train_config, session, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  
        
        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')         
        trainer = ModelTraining(self.configuration)           
        trainer.set_device()

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = train_config.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        splitter = TrainValidationSplit(train_config) 
        train_data, validation_data = splitter.split_train_and_validation(images_paths)     

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        builder = TrainingDataLoader(train_config)           
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, validation_data)        
                            
        # resume training from pretrained model    
        logger.info('Resuming XREPORT AutoEncoder training from checkpoint') 
        trainer.resume_training(
            model, train_dataset, validation_dataset, checkpoint_path, session,
            progress_callback=progress_callback)
        
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   



###############################################################################
class InferenceEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)        
        self.modser = ModelSerializer()         
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        return self.modser.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, device='CPU', progress_callback=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')         
        model, train_config, session, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        trainer = ModelTraining(self.configuration)    
        trainer.set_device(device_override=device)

        # select images from the inference folder and retrieve current paths        
        images_paths = self.serializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'{len(images_paths)} images have been found as inference input')       
        # extract features from images using the encoder output, the image encoder
        # takes the list of images path from inference as input    
        encoder = ImageEncoding(model, self.configuration, checkpoint_path)  
        logger.info(f'Start encoding images using model {selected_checkpoint}')  
        encoder.encode_images_features(images_paths, progress_callback) 
        logger.info('Encoded images have been saved as .npy')
           
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   



