import cv2
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from XREPORT.commons.utils.validation.dataset import ImageAnalysis, TextAnalysis
from XREPORT.commons.utils.data.loader import TrainingDataLoader
from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.learning.training import ModelTraining
from XREPORT.commons.utils.validation.reports import log_training_report
from XREPORT.commons.constants import IMG_PATH, INFERENCE_INPUT_PATH
from XREPORT.commons.logger import logger



###############################################################################
class ValidationEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)            
        self.img_analyzer = ImageAnalysis(configuration) 
        self.text_analyzer = TextAnalysis()
        self.text_placeholder = "No description available for this image."    
        self.configuration = configuration  

    #--------------------------------------------------------------------------
    def load_images_path(self, path, sample_size=1.0):        
        images_paths = self.serializer.get_images_path_from_directory(
            path, sample_size) 
        
        return images_paths 
    
    #--------------------------------------------------------------------------
    def get_description_from_image(self, image_name):               
        dataset = self.serializer.load_source_dataset()
        image_name = image_name.split('.')[0]        
        description = dataset.loc[dataset['image'] == image_name, 'text'].values
        description = description[0] if len(description) > 0 else self.text_placeholder  
        
        return description
        
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None, worker=None):
        sample_size = self.configuration.get("sample_size", 1.0)
        dataset = self.serializer.load_source_dataset(sample_size)   
        dataset = self.serializer.update_images_path(dataset)
        logger.info(f'Selected sample size for dataset evaluation: {sample_size}')
        logger.info(f'Number of reports and related images: {dataset.shape[0]}')
        
        words = self.text_analyzer.count_words_in_documents(dataset)
        logger.info(f'Number of words dataset: {len(words)}')
        logger.info(f'Number of unique words dataset: {len(set(words))}')   
            
        
        images = []        
        if 'image_stats' in metrics:
            logger.info('Current metric: image dataset statistics')
            image_statistics = self.img_analyzer.calculate_image_statistics(
                dataset, progress_callback, worker)                      
             
        if 'pixels_distribution' in metrics:
            logger.info('Current metric: pixel intensity distribution')
            images.append(self.img_analyzer.calculate_pixel_intensity_distribution(
                dataset, progress_callback, worker))

        return images 

    #--------------------------------------------------------------------------
    def get_checkpoints_summary(self, progress_callback=None, worker=None): 
        summarizer = ModelEvaluationSummary(self.configuration)    
        checkpoints_summary = summarizer.get_checkpoints_summary(progress_callback, worker) 
        logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')   
    
    #--------------------------------------------------------------------------
    def run_model_evaluation_pipeline(self, metrics, selected_checkpoint, device='CPU', 
                                      progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')   
        modser = ModelSerializer()       
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')                
        trainer = ModelTraining(train_config)    
        trainer.set_device(device_override=device)  

        # isolate the encoder from the autoencoder model   
        encoder = ImageEncoding(model, train_config, checkpoint_path)
        encoder_model = encoder.encoder_model 

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = train_config.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        splitter = TrainValidationSplit(train_config) 
        _, validation_images = splitter.split_train_and_validation(images_paths)     

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        # use tf.data.Dataset to build the model dataloader with a larger batch size
        # the dataset is built on top of the training and validation data
        loader = InferenceDataLoader(train_config)    
        validation_dataset = loader.build_inference_dataloader(
            validation_images, batch_size=1)              

        images = []
        if 'evaluation_report' in metrics:
            # evaluate model performance over the training and validation dataset 
            summarizer = ModelEvaluationSummary(self.configuration)       
            summarizer.evaluation_report(model, validation_dataset, worker=worker) 

        if 'image_reconstruction' in metrics:
            validator = ImageReconstruction(train_config, model, checkpoint_path)      
            images.append(validator.visualize_reconstructed_images(
                validation_images, progress_callback, worker=worker))       

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
    def run_training_pipeline(self, progress_callback=None, worker=None):  
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
        logger.info('Building FeXT AutoEncoder model based on user configuration') 
        checkpoint_path = self.modser.create_checkpoint_folder()
        autoencoder = FeXTAutoEncoder(self.configuration)           
        model = autoencoder.get_model(model_summary=True) 

        # generate training log report and graphviz plot for the model layout         
        log_training_report(train_data, validation_data, self.configuration)        
        self.modser.save_model_plot(model, checkpoint_path) 
        # perform training and save model at the end
        logger.info('Starting FeXT AutoEncoder training') 
        trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, 
            progress_callback=progress_callback, worker=worker)
        
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None, 
                                 worker=None):
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
        logger.info('Resuming FeXT AutoEncoder training from checkpoint') 
        trainer.resume_training(
            model, train_dataset, validation_dataset, checkpoint_path, session,
            progress_callback=progress_callback, worker=worker)
        
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
    def run_inference_pipeline(self, selected_checkpoint, device='CPU', 
                               progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')         
        model, train_config, session, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        trainer = ModelTraining(train_config)    
        trainer.set_device(device_override=device)

        # select images from the inference folder and retrieve current paths        
        images_paths = self.serializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'{len(images_paths)} images have been found as inference input')       
        # extract features from images using the encoder output, the image encoder
        # takes the list of images path from inference as input    
        encoder = ImageEncoding(model, train_config, checkpoint_path)  
        logger.info(f'Start encoding images using model {selected_checkpoint}')  
        encoder.encode_images_features(images_paths, progress_callback, worker=worker) 
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

   



