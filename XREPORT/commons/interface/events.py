import os
import cv2
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from XREPORT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.commons.utils.validation.dataset import ImageAnalysis, TextAnalysis
from XREPORT.commons.utils.validation.checkpoints import ModelEvaluationSummary, EvaluateTextConsistency
from XREPORT.commons.utils.data.process import TextSanitizer, TrainValidationSplit, TokenWizard
from XREPORT.commons.utils.data.loader import XRAYDataLoader
from XREPORT.commons.utils.learning.training.fitting import ModelTraining
from XREPORT.commons.utils.learning.models.transformers import XREPORTModel
from XREPORT.commons.utils.learning.inference.generator import TextGenerator
from XREPORT.commons.interface.workers import check_thread_status

from XREPORT.commons.constants import INFERENCE_INPUT_PATH
from XREPORT.commons.logger import logger


###############################################################################
class GraphicsHandler:

    def __init__(self): 
        self.image_encoding = cv2.IMREAD_UNCHANGED
        self.gray_scale_encoding = cv2.IMREAD_GRAYSCALE
        self.BGRA_encoding = cv2.COLOR_BGRA2RGBA
        self.BGR_encoding = cv2.COLOR_BGR2RGB

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
        img = cv2.imread(path, self.image_encoding)
        # Handle grayscale, RGB, or RGBA
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, self.gray_scale_encoding)
        elif img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, self.BGRA_encoding)
        else:  # BGR
            img = cv2.cvtColor(img, self.BGR_encoding)

        h, w = img.shape[:2]
        if img.shape[2] == 3:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        else:  
            qimg = QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)
    

###############################################################################
class DatasetEvents:

    def __init__(self, database, configuration):        
        self.text_placeholder = "No description available for this image."   
        self.database = database             
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def load_images_path(self, path, sample_size=1.0):
        serializer = DataSerializer(self.database, self.configuration)         
        images_paths = serializer.get_images_path_from_directory(
            path, sample_size) 
        
        return images_paths 
    
    #--------------------------------------------------------------------------
    def get_description_from_train_image(self, image_name : str):    
        serializer = DataSerializer(self.database, self.configuration)             
        dataset = serializer.load_source_dataset(sample_size=1.0)
        image_no_ext = image_name.split('.')[0]  
        mask = dataset['image'].astype(str).str.contains(image_no_ext, case=False, na=False)
        description = dataset.loc[mask, 'text'].values
        description = description[0] if len(description) > 0 else self.text_placeholder  
        
        return description 

    #--------------------------------------------------------------------------
    def get_generated_report(self, image_name : str):
        serializer = DataSerializer(self.database, self.configuration)                 
        dataset = serializer.load_source_dataset(sample_size=1.0)
        image_no_ext = image_name.split('.')[0]  
        mask = dataset['image'].astype(str).str.contains(image_no_ext, case=False, na=False)
        description = dataset.loc[mask, 'text'].values
        description = description[0] if len(description) > 0 else self.text_placeholder  
        
        return description   
    
    #--------------------------------------------------------------------------
    def run_dataset_builder(self, progress_callback=None, worker=None):
        serializer = DataSerializer(self.database, self.configuration)      
        sample_size = self.configuration.get("sample_size", 1.0)            
        dataset = serializer.load_source_dataset(sample_size=sample_size)
        
        # sanitize text corpus by removing undesired symbols and punctuation     
        sanitizer = TextSanitizer(self.configuration)
        processed_data = sanitizer.sanitize_text(dataset)
        logger.info(f'Dataset includes {processed_data.shape[0]} samples')

        # preprocess text corpus using selected pretrained tokenizer. Text is tokenized
        # into subunits and these are eventually mapped to integer indexes        
        tokenization = TokenWizard(self.configuration) 
        logger.info(f'Tokenizing text corpus using pretrained {tokenization.tokenizer_name} tokenizer')    
        processed_data = tokenization.tokenize_text_corpus(processed_data)   
        vocabulary_size = tokenization.vocabulary_size 
        logger.info(f'Vocabulary size (unique tokens): {vocabulary_size}')
        
        # split data into train set and validation set
        logger.info('Preparing dataset of images and captions based on splitting size')  
        splitter = TrainValidationSplit(self.configuration, processed_data)     
        train_data, validation_data = splitter.split_train_and_validation()        
               
        self.serializer.save_train_and_validation_data(
            train_data, validation_data, vocabulary_size) 
        logger.info('Preprocessed data saved into XREPORT database')              



###############################################################################
class ValidationEvents:

    def __init__(self, database, configuration):
        self.database = database 
        self.configuration = configuration  
            
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None, worker=None):
        serializer = DataSerializer(self.database, self.configuration) 
        sample_size = self.configuration.get("sample_size", 1.0)
        dataset = serializer.load_source_dataset(sample_size)   
        dataset = serializer.update_images_path(dataset)
        logger.info(f'Selected sample size for dataset evaluation: {sample_size}')
        logger.info(f'Number of reports and related images: {dataset.shape[0]}')

        img_analyzer = ImageAnalysis(self.database, self.configuration)
        text_analyzer = TextAnalysis(self.database, self.configuration)                
       
        # 1. calculate images statistics 
        if 'image_statistics' in metrics:            
            logger.info('Current metric: image dataset statistics')
            image_statistics = img_analyzer.calculate_image_statistics(
                dataset, progress_callback=progress_callback, worker=worker)  

        # 2. calculate text statistics 
        if 'text_statistics' in metrics:
            logger.info('Current metric: text dataset statistics')            
            text_statistics = text_analyzer.calculate_text_statistics(
                dataset, progress_callback=progress_callback, worker=worker)  
                                
        images = []            
        if 'pixels_distribution' in metrics:
            logger.info('Current metric: pixel intensity distribution')
            images.append(img_analyzer.calculate_pixel_intensity_distribution(
                dataset, progress_callback=progress_callback, worker=worker)) 

        return images 

    #--------------------------------------------------------------------------
    def get_checkpoints_summary(self, progress_callback=None, worker=None):
        summarizer = ModelEvaluationSummary(self.configuration)    
        checkpoints_summary = summarizer.get_checkpoints_summary(
            progress_callback=progress_callback, worker=worker)  
 
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
        
        serializer = DataSerializer(self.database, self.configuration) 
        _, val_data, metadata = serializer.load_train_and_validation_data()    
        val_data = serializer.update_images_path(val_data)  
        vocabulary_size = metadata['vocabulary_size']
        logger.info(f'Validation data has been loaded: {val_data.shape[0]} samples')    
        logger.info(f'Vocabolary size: {vocabulary_size} tokens')    

        loader = XRAYDataLoader(self.configuration)      
        validation_dataset = loader.build_inference_dataloader(val_data)

        # check worker status to allow interruption
        check_thread_status(worker)

        images = []
        if 'evaluation_report' in metrics:
            # evaluate model performance over the training and validation dataset 
            summarizer = ModelEvaluationSummary(self.configuration)       
            summarizer.get_evaluation_report(
                model, validation_dataset, 
                progress_callback=progress_callback, worker=worker)  

        if 'BLEU_score' in metrics:            
            # One can select different either greedy search or beam search to genarate
            # reports with a pretrained decoder        
            scoring = EvaluateTextConsistency(model, self.configuration)
            scores = scoring.calculate_BLEU_score(
                val_data, progress_callback=progress_callback, worker=worker)  
          

        return images  
    
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(exc, '\n', tb)
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   

###############################################################################
class ModelEvents:

    def __init__(self, database, configuration): 
        self.database = database  
        self.configuration = configuration 
    
    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        serializer = ModelSerializer()
        return serializer.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, device='CPU', 
                               progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')
        modser = ModelSerializer()         
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        trainer = ModelTraining(train_config)    
        trainer.set_device(device_override=device)

        # select images from the inference folder and retrieve current paths    
        serializer = DataSerializer(self.database, self.configuration)      
        img_paths = serializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'\nStart generating reports using model {os.path.basename(checkpoint_path)}')
        logger.info(f'{len(img_paths)} images have been found and are ready for inference pipeline')

        # check worker status to allow interruption
        check_thread_status(worker)

        # generate radiological reports from the list of inference image paths 
        inference_mode = self.configuration.get("inference_mode", 'greedy_search')  
        generator = TextGenerator(model, train_config, inference_mode, checkpoint_path) 
        generated_reports = generator.generate_radiological_reports(
            img_paths, progress_callback=progress_callback, worker=worker)
        serializer.save_generated_reports(generated_reports)
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None, worker=None):
        serializer = DataSerializer(self.database, self.configuration)    
        train_data, validation_data, metadata = serializer.load_train_and_validation_data() 
        # fetch images path from the preprocessed data
        train_data = serializer.update_images_path(train_data)
        validation_data = serializer.update_images_path(validation_data)
        vocabulary_size = metadata['vocabulary_size'] 
        
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        builder = XRAYDataLoader(self.configuration)   
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, validation_data) 
        
        # set device for training operations based on user configuration        
        logger.info('Setting device for training operations based on user configuration') 
        trainer = ModelTraining(self.configuration)           
        trainer.set_device()

        # build the autoencoder model 
        logger.info('Building FeXT AutoEncoder model based on user configuration') 
        modser = ModelSerializer() 
        checkpoint_path = modser.create_checkpoint_folder()

        # check worker status to allow interruption
        check_thread_status(worker)

        # initialize and compile the captioning model    
        logger.info('Building XREPORT Transformer model')
        captioner = XREPORTModel(vocabulary_size, self.configuration)
        model = captioner.get_model(model_summary=True) 

        # generate training log report and graphviz plot for the model layout               
        modser.save_model_plot(model, checkpoint_path)
        
        logger.info('Starting XREPORT Transformer model training') 
        trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, 
            progress_callback=progress_callback, worker=worker)
                
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')
        modser = ModelSerializer()         
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  
        
        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')         
        trainer = ModelTraining(self.configuration)           
        trainer.set_device()
        
        serializer = DataSerializer(self.database, self.configuration)
        train_data, validation_data, metadata = serializer.load_train_and_validation_data()
        train_data = serializer.update_images_path(train_data)
        validation_data = serializer.update_images_path(validation_data)       
        
        # create the tf.datasets using the previously initialized generators 
        builder = TrainingDataLoader(self.configuration)   
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, validation_data)     

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        builder = TrainingDataLoader(train_config)           
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, validation_data)        
                            
        # resume training from pretrained model    
        logger.info(f'Resuming training from checkpoint {selected_checkpoint}') 
        trainer.resume_training(
            model, train_dataset, validation_dataset, checkpoint_path, session,
            progress_callback=progress_callback, worker=worker)  

    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, device='CPU', 
                               progress_callback=None, worker=None):
        modser = ModelSerializer() 
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')         
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        trainer = ModelTraining(train_config)    
        trainer.set_device(device_override=device)

        # select images from the inference folder and retrieve current paths
        serializer = DataSerializer(self.database, self.configuration)        
        img_paths = serializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'\nStart generating reports using model {os.path.basename(checkpoint_path)}')
        logger.info(f'{len(img_paths)} images have been found and are ready for inference pipeline')

        # generate radiological reports from the list of inference image paths 
        inference_mode = self.configuration.get("inference_mode", 'greedy_search')  
        generator = TextGenerator(model, train_config, inference_mode, checkpoint_path) 
        generated_reports = generator.generate_radiological_reports(
            img_paths, progress_callback=progress_callback, worker=worker)
        serializer.save_generated_reports(generated_reports)                 
        
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(exc, '\n', tb)
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

