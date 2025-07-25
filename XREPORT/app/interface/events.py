import os
import cv2

from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from XREPORT.app.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.app.utils.validation.dataset import ImageAnalysis, TextAnalysis
from XREPORT.app.utils.validation.checkpoints import ModelEvaluationSummary, EvaluateTextQuality
from XREPORT.app.utils.data.process import TextSanitizer, TrainValidationSplit, TokenizerHandler
from XREPORT.app.utils.data.loader import XRAYDataLoader
from XREPORT.app.utils.learning.device import DeviceConfig
from XREPORT.app.utils.learning.training.fitting import ModelTraining
from XREPORT.app.utils.learning.models.transformers import XREPORTModel
from XREPORT.app.utils.learning.inference.generator import TextGenerator
from XREPORT.app.interface.workers import check_thread_status, update_progress_callback

from XREPORT.app.constants import INFERENCE_INPUT_PATH
from XREPORT.app.logger import logger


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
        if img is None:
            return 

        # Convert to RGB or RGBA as needed
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, self.gray_scale_encoding)
            qimg_format = QImage.Format_RGB888
            channels = 3
        elif img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, self.BGRA_encoding)
            qimg_format = QImage.Format_RGBA8888
            channels = 4
        else:  # BGR
            img = cv2.cvtColor(img, self.BGR_encoding)
            qimg_format = QImage.Format_RGB888
            channels = 3

        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, channels * w, qimg_format)
        return QPixmap.fromImage(qimg)
    

###############################################################################
class DatasetEvents:

    def __init__(self, configuration : dict):
        self.serializer = DataSerializer(configuration)
        self.full_dataset = self.serializer.load_source_dataset(sample_size=1.0)        
        self.text_placeholder = "No description available for this image." 
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def load_images_path(self, path, sample_size=1.0):
        img_paths = self.serializer.get_images_path_from_directory(path, sample_size) 
        
        return img_paths 
    
    #--------------------------------------------------------------------------
    def get_description_from_image(self, image_name : str):
        image_no_ext = image_name.split('.')[0]  
        mask = self.full_dataset['image'].astype(str).str.contains(image_no_ext, case=False, na=False)
        description = self.full_dataset.loc[mask, 'text'].values
        description = description[0] if len(description) > 0 else self.text_placeholder  
        
        return description 
    
    #--------------------------------------------------------------------------
    def run_dataset_builder(self, progress_callback=None, worker=None):
        sample_size = self.configuration.get("sample_size", 1.0)            
        dataset = self.serializer.load_source_dataset(sample_size=sample_size)
        if dataset is None or dataset.empty:
            logger.error("No data found in the database")
            return          

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(1, 4, progress_callback)
        
        # sanitize text corpus by removing undesired symbols and punctuation     
        sanitizer = TextSanitizer(self.configuration)
        processed_data = sanitizer.sanitize_text(dataset)
        logger.info(f'Dataset includes {processed_data.shape[0]} samples')

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(2, 4, progress_callback)

        # preprocess text corpus using selected pretrained tokenizer. Text is tokenized
        # into subunits and these are eventually mapped to integer indexes        
        tokenization = TokenizerHandler(self.configuration) 
        logger.info(f'Tokenizing text corpus using pretrained {tokenization.tokenizer_id} tokenizer')    
        processed_data = tokenization.tokenize_text_corpus(processed_data)   
        vocabulary_size = tokenization.vocabulary_size 
        logger.info(f'Vocabulary size (unique tokens): {vocabulary_size}')

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(3, 4, progress_callback)
        
        # drop raw text columns and keep only tokenized text 
        processed_data = processed_data.drop(columns=['text']) 
        # split data into train set and validation set
        logger.info('Preparing dataset of images and captions based on splitting size')
        splitter = TrainValidationSplit(self.configuration, processed_data)     
        train_data, validation_data = splitter.split_train_and_validation()
        # save preprocessed data into database
        self.serializer.save_train_and_validation_data(train_data, validation_data, vocabulary_size) 
        logger.info('Preprocessed data saved into database') 

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(4, 4, progress_callback)

    #--------------------------------------------------------------------------
    @staticmethod
    def rebuild_dataset_from_metadata(metadata : dict):
        serializer = DataSerializer(metadata)
        sample_size = metadata.get("sample_size", 1.0)            
        dataset = serializer.load_source_dataset(sample_size=sample_size)
        if dataset is None or dataset.empty:
            logger.error("No data found in the database during dataset rebuilding")
            return   
        
        # sanitize text corpus by removing undesired symbols and punctuation     
        sanitizer = TextSanitizer(metadata)
        processed_data = sanitizer.sanitize_text(dataset)

        # preprocess text corpus using selected pretrained tokenizer. Text is tokenized
        # into subunits and these are eventually mapped to integer indexes        
        tokenization = TokenizerHandler(metadata) 
        logger.info(f'Tokenizing text corpus using pretrained {tokenization.tokenizer_id} tokenizer')    
        processed_data = tokenization.tokenize_text_corpus(processed_data)   
        vocabulary_size = tokenization.vocabulary_size 
        logger.info(f'Vocabulary size (unique tokens): {vocabulary_size}')
        
        # split data into train set and validation set           
        splitter = TrainValidationSplit(metadata, processed_data)     
        train_data, validation_data = splitter.split_train_and_validation()

        return train_data, validation_data 

    

###############################################################################
class ValidationEvents:

    def __init__(self, configuration : dict):         
        self.configuration = configuration  
            
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None, worker=None):
        serializer = DataSerializer(self.configuration) 
        sample_size = self.configuration.get("sample_size", 1.0)
        dataset = serializer.load_source_dataset(sample_size)   
        dataset = serializer.update_images_path(dataset)        
        logger.info(f'Selected reports and related images: {len(dataset)}')

        img_analyzer = ImageAnalysis(self.configuration)
        text_analyzer = TextAnalysis(self.configuration)                
       
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
    def run_model_evaluation_pipeline(self, metrics, selected_checkpoint, progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint')   
        modser = ModelSerializer()       
        model, train_config, train_metadata, _, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # set device for training operations
        logger.info('Setting device for training operations')                
        device = DeviceConfig(self.configuration)   
        device.set_device()   
        # load validation data and current preprocessing metadata. This must
        # be compatible with the currently loaded checkpoint configurations
        serializer = DataSerializer(train_config) 
        current_metadata = serializer.load_train_and_validation_data(only_metadata=True)
        validated_metadata = serializer.validate_metadata(current_metadata, train_metadata)
        # just load the data if metadata is compatible
        if validated_metadata:
            logger.info('Loading processed dataset as it is compatible with the selected checkpoint')
            _, validation_data, train_metadata = serializer.load_train_and_validation_data()
        else:     
            logger.info(f'Rebuilding dataset from {selected_checkpoint} metadata')
            _, validation_data = DatasetEvents.rebuild_dataset_from_metadata(train_metadata)
        # update image paths in the validation data using currently available images
        validation_data = serializer.update_images_path(validation_data)

        vocabulary_size = train_metadata.get('vocabulary_size', 1000)
        logger.info(f'Validation data has been loaded: {len(validation_data)} samples')    
        logger.info(f'Vocabolary size: {vocabulary_size} tokens')          
        num_samples = self.configuration.get('num_evaluation_samples', 10)
        loader = XRAYDataLoader(train_config)
        validation_dataset = loader.build_training_dataloader(validation_data)     

        # check worker status to allow interruption
        check_thread_status(worker)

        images = []
        if 'evaluation_report' in metrics:            
            # evaluate model performance over the validation dataset 
            logger.info('Current metric: model loss and metrics evaluation')
            summarizer = ModelEvaluationSummary(self.configuration) 
            summarizer.get_evaluation_report(model, validation_dataset, worker=worker)   

        if 'BLEU_score' in metrics:
            logger.info('Current metric: BLEU score')
            scoring = EvaluateTextQuality(
                model, train_config, train_metadata, num_samples)            
            scores = scoring.calculate_BLEU_score(
                validation_data, progress_callback=progress_callback, worker=worker)  
          
        return images  
    
    
###############################################################################
class ModelEvents:

    def __init__(self, configuration): 
        self.configuration = configuration 
    
    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        serializer = ModelSerializer()
        return serializer.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None, worker=None):
        serializer = DataSerializer(self.configuration)    
        train_data, validation_data, metadata = serializer.load_train_and_validation_data()
        if train_data.empty or validation_data.empty:
            logger.warning("No data found in the database for training")
            return

        # fetch images path from the preprocessed data
        train_data = serializer.update_images_path(train_data)
        validation_data = serializer.update_images_path(validation_data)
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        builder = XRAYDataLoader(self.configuration)   
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)

        # check worker status to allow interruption
        check_thread_status(worker)

        # set device for training operations        
        logger.info('Setting device for training operations') 
        device = DeviceConfig(self.configuration)   
        device.set_device() 
        # create checkpoint folder     
        modser = ModelSerializer() 
        checkpoint_path = modser.create_checkpoint_folder()
        # initialize and compile the captioning model    
        logger.info('Building XREPORT Transformer model')
        captioner = XREPORTModel(metadata, self.configuration)
        model = captioner.get_model(model_summary=True) 
        # generate training log report and graphviz plot for the model layout               
        modser.save_model_plot(model, checkpoint_path)
        # start model training
        logger.info('Starting XREPORT Transformer model training')
        trainer = ModelTraining(self.configuration)
        trainer.train_model(
            model, train_dataset, validation_dataset, metadata, checkpoint_path, 
            progress_callback=progress_callback, worker=worker) 
                
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint')
        modser = ModelSerializer()         
        model, train_config, train_metadata, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)
        # set device for training operations
        logger.info('Setting device for training operations')         
        device = DeviceConfig(self.configuration)   
        device.set_device() 

        # check worker status to allow interruption
        check_thread_status(worker)

        # load metadata and check whether this is compatible with the current checkpoint
        # rebuild dataset if metadata is not compatible and the user has requested this feature
        serializer = DataSerializer(train_config)
        current_metadata = serializer.load_train_and_validation_data(only_metadata=True)
        validated_metadata = serializer.validate_metadata(current_metadata, train_metadata)
        # just load the data if metadata is compatible
        if validated_metadata:
            logger.info('Loading processed dataset as it is compatible with the selected checkpoint')
            train_data, validation_data, train_metadata = serializer.load_train_and_validation_data()
        else:     
            logger.info(f'Rebuilding dataset from {selected_checkpoint} metadata')
            train_data, validation_data = DatasetEvents.rebuild_dataset_from_metadata(train_metadata)

        # update image paths in the train and validation data using currently available images
        train_data = serializer.update_images_path(train_data)
        validation_data = serializer.update_images_path(validation_data) 
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        builder = XRAYDataLoader(train_config)           
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)
        
        # check worker status to allow interruption
        check_thread_status(worker)

        # resume training from pretrained model    
        logger.info(f'Resuming training from checkpoint {selected_checkpoint}') 
        additional_epochs = self.configuration.get('additional_epochs', 100)
        trainer = ModelTraining(train_config, train_metadata)
        trainer.resume_training(
            model, train_dataset, validation_dataset, train_metadata, checkpoint_path, session,
            additional_epochs, progress_callback=progress_callback, worker=worker)  

    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, progress_callback=None, worker=None):
        modser = ModelSerializer() 
        logger.info(f'Loading {selected_checkpoint} checkpoint')         
        model, train_config, train_metadata, _, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        device = DeviceConfig(self.configuration)   
        device.set_device()
        # select images from the inference folder and retrieve current paths
        serializer = DataSerializer(self.configuration)        
        img_paths = serializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'Start generating reports using model {os.path.basename(checkpoint_path)}')
        logger.info(f'{len(img_paths)} images have been found and are ready for inference pipeline')
        # generate radiological reports from the list of inference image paths 
        inference_mode = self.configuration.get("inference_mode", 'greedy_search') 
        max_report_size = train_metadata.get('max_report_size', 200) 
        generator = TextGenerator(model, train_config, max_report_size) 
        generated_reports = generator.generate_radiological_reports(
            img_paths, inference_mode, 
            progress_callback=progress_callback, worker=worker)
        
        # package inference outputs to fit the database table
        reports = [{'image' : os.path.basename(k), 
                    'report': v,
                    'checkpoint': selected_checkpoint} 
                    for k, v in generated_reports.items()]
        
        serializer.save_generated_reports(reports) 
        logger.info('Generated reports have been saved in the database')                
        
   
