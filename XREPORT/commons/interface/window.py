from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

import os
from functools import partial

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView, QPlainTextEdit)

from XREPORT.commons.utils.data.database import XREPORTDatabase
from XREPORT.commons.configuration import Configuration
from XREPORT.commons.interface.workers import Worker
from XREPORT.commons.interface.events import (GraphicsHandler, DatasetEvents, ValidationEvents, 
                                              ModelEvents)

from XREPORT.commons.constants import IMG_PATH, INFERENCE_INPUT_PATH
from XREPORT.commons.logger import logger



###############################################################################
class MainWindow:
    
    def __init__(self, ui_file_path: str): 
        super().__init__()           
        loader = QUiLoader()
        ui_file = QFile(ui_file_path)
        ui_file.open(QIODevice.ReadOnly)
        self.main_win = loader.load(ui_file)
        ui_file.close() 

        # Checkpoint & metrics state
        self.selected_checkpoint = None
        self.selected_metrics = {'dataset': [], 'model': []}       
          
        # initial settings
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()
    
        self.threadpool = QThreadPool.globalInstance()
        self.worker = None
        self.worker_running = False     

        # initialize database
        self.database = XREPORTDatabase(self.configuration)
        self.database.initialize_database()  
        self.database.update_database()            

        # --- Create persistent handlers ---
        self.graphic_handler = GraphicsHandler()
        self.dataset_handler = DatasetEvents(self.database, self.configuration)
        self.validation_handler = ValidationEvents(self.database, self.configuration)
        self.model_handler = ModelEvents(self.database, self.configuration)        

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([ 
            (QPushButton,'stopThread','stop_thread'),  
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QProgressBar,'progressBar','progress_bar'),         
            # 1. dataset tab page
            (QCheckBox,'getStatsAnalysis','get_image_stats'),
            (QCheckBox,'getPixDist','pixel_distribution_metric'),
            (QPushButton,'getImgMetrics','get_img_metrics'),
            (QSpinBox,'seed','general_seed'),
            (QDoubleSpinBox,'sampleSize','sample_size'),
            (QSpinBox,'maxReportSize','max_report_size'), 
            (QComboBox,'tokenizerList','tokenizer'),
            (QPushButton,'buildMLDataset','build_training_dataset'),           
                      
            # 2. training tab page    
            (QCheckBox,'imgAugment','img_augmentation'),
            (QCheckBox,'setShuffle','use_shuffle'),
            (QDoubleSpinBox,'trainSampleSize','train_sample_size'),            
            (QDoubleSpinBox,'validationSize','validation_size'),
            (QSpinBox,'shuffleSize','shuffle_size'),            
            (QRadioButton,'setCPU','use_CPU'),
            (QRadioButton,'setGPU','use_GPU'),
            (QSpinBox,'deviceID','device_ID'),
            (QSpinBox,'numWorkers','num_workers'),
            (QCheckBox,'runTensorboard','use_tensorboard'),
            (QCheckBox,'realTimeHistory','get_real_time_history'),
            (QCheckBox,'saveCheckpoints','save_checkpoints'),
            (QSpinBox,'trainSeed','train_seed'),
            (QSpinBox,'splitSeed','split_seed'),
            (QSpinBox,'numEpochs','epochs'),
            (QSpinBox,'batchSize','batch_size'),            
            (QSpinBox,'saveCPFrequency','save_cp_frequency'),
            (QCheckBox,'useScheduler','LR_scheduler'), 
            (QDoubleSpinBox,'postWarmLR','post_warmup_LR'),
            (QDoubleSpinBox,'warmUpSteps','warmup_steps'),            
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'), 
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),   
            (QComboBox,'backendJIT','jit_backend'), 
            (QCheckBox,'freezeImgEncoder','freeze_img_encoder'),          
            (QSpinBox,'attentionHeads','num_attention_heads'),
            (QSpinBox,'numEncoders','num_encoders'),                   
            (QSpinBox,'numDecoders','num_decoders'),
            (QSpinBox,'embeddingDims','embedding_dimensions'),
            (QDoubleSpinBox,'trainTemp','train_temperature'),         
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),
            (QComboBox,'checkpointsList','checkpoints_list'),            
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            # 3. model evaluation tab page
            (QPushButton,'evaluateModel','model_evaluation'),
            (QCheckBox,'runEvaluationGPU','use_GPU_evaluation'), 
            (QPushButton,'checkpointSummary','checkpoints_summary'),
            (QCheckBox,'evalReport','get_evaluation_report'), 
            (QCheckBox,'getBLEUScore','get_BLEU_score'),      
            (QSpinBox,'numImages','num_evaluation_images'),           
            # 4. inference tab page  
            (QDoubleSpinBox,'inferenceTemp','inference_temperature'),
            (QComboBox,'inferenceMode','inference_mode'),  
            (QCheckBox,'runInferenceGPU','use_GPU_inference'),      
            (QPushButton,'generateReports','generate_reports'),          
            # 5. Viewer tab
            (QPushButton,'loadImages','load_source_images'),
            (QPushButton,'previousImg','previous_image'),
            (QPushButton,'nextImg','next_image'),
            (QPushButton,'clearImg','clear_images'),
            (QRadioButton,'viewDataPlots','data_plots_view'),
            (QRadioButton,'viewEvalPlots','model_plots_view'),
            (QRadioButton,'viewInferenceImages','inference_images_view'),
            (QRadioButton,'viewTrainImages','train_images_view'), 
            (QPlainTextEdit, 'description', 'image_description'),           
            ])
        
        self._connect_signals([  
            ('checkpoints_list','currentTextChanged',self.select_checkpoint), 
            ('refresh_checkpoints','clicked',self.load_checkpoints),
            ('stop_thread','clicked',self.stop_running_worker),          
            # 1. dataset tab page            
            ('pixel_distribution_metric','toggled',self._update_metrics),
            ('get_img_metrics','clicked',self.run_dataset_evaluation_pipeline), 
            ('build_ML_dataset','clicked',self.build_ML_dataset),
            # 2. training tab page                                   
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),
            # 3. model evaluation tab page            
            ('get_evaluation_report','toggled',self._update_metrics), 
            ('get_BLEU_score','toggled',self._update_metrics),
            ('model_evaluation','clicked', self.run_model_evaluation_pipeline),
            ('checkpoints_summary','clicked',self.get_checkpoints_summary),            
            # 4. inference tab page  
            ('generate_reports','clicked',self.generate_reports_with_checkpoint),            
            # 5. viewer tab page 
            ('data_plots_view', 'toggled', self._update_graphics_view),
            ('model_plots_view', 'toggled', self._update_graphics_view),
            ('inference_images_view', 'toggled', self._update_graphics_view), 
            ('train_images_view', 'toggled', self._update_graphics_view), 
            ('load_source_images','clicked', self.load_images),
            ('previous_image', 'clicked', self.show_previous_figure),
            ('next_image', 'clicked', self.show_next_figure),
            ('clear_images', 'clicked', self.clear_figures),                        
        ]) 

        self.pixmap_source_map = {
            self.data_plots_view: ("dataset_eval_images", "dataset_eval_images"),
            self.model_plots_view: ("model_eval_images", "model_eval_images"),
            self.inference_images_view: ("inference_images", "inference_images"),
            self.train_images_view: ("train_images", "train_images")} 
        
        self._auto_connect_settings() 
        self.use_GPU.toggled.connect(self._update_device)
        self.use_CPU.toggled.connect(self._update_device)
        
        # Initial population of dynamic UI elements
        self.load_checkpoints()
        self._set_graphics()         

    # [SHOW WINDOW]
    ###########################################################################
    def show(self):        
        self.main_win.show()           

    # [HELPERS]
    ###########################################################################
    def connect_update_setting(self, widget, signal_name, config_key, getter=None):
        if getter is None:
            if isinstance(widget, (QCheckBox, QRadioButton)):
                getter = widget.isChecked
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                getter = widget.value
            elif isinstance(widget, QComboBox):
                getter = widget.currentText
           
        signal = getattr(widget, signal_name)
        signal.connect(partial(self._update_single_setting, config_key, getter))

    #--------------------------------------------------------------------------
    def _update_single_setting(self, config_key, getter, *args):
        value = getter()
        self.config_manager.update_value(config_key, value)

    #--------------------------------------------------------------------------
    def _auto_connect_settings(self):
        connections = [
            # 1. dataset tab page
            ('general_seed', 'valueChanged', 'general_seed'),
            ('sample_size', 'valueChanged', 'sample_size'),
            ('max_report_size', 'valueChanged', 'max_report_size'),
            ('tokenizer', 'currentTextChanged', 'tokenizer'),
            # 2. training tab page               
            ('img_augmentation', 'toggled', 'img_augmentation'),
            ('use_shuffle', 'toggled', 'shuffle_dataset'),
            ('num_workers', 'valueChanged', 'num_workers'),
            ('use_mixed_precision', 'toggled', 'mixed_precision'),
            ('freeze_img_encoder', 'toggled', 'freeze_img_encoder'),            
            ('num_attention_heads', 'valueChanged', 'num_attention_heads'),
            ('num_encoders', 'valueChanged', 'num_encoders'),
            ('num_decoders', 'valueChanged', 'num_decoders'),
            ('embedding_dimensions', 'valueChanged', 'embedding_dimensions'),
            ('train_temperature', 'valueChanged', 'embedding_dimensions'),
            ('use_JIT_compiler', 'toggled', 'use_jit_compiler'),
            ('jit_backend', 'currentTextChanged', 'jit_backend'),
            ('use_tensorboard', 'toggled', 'run_tensorboard'),
            ('get_real_time_history', 'toggled', 'real_time_history'),
            ('save_checkpoints', 'toggled', 'save_checkpoints'),
            ('LR_scheduler', 'toggled', 'use_lr_scheduler'),
            ('split_seed', 'valueChanged', 'split_seed'),
            ('train_seed', 'valueChanged', 'train_seed'),
            ('shuffle_size', 'valueChanged', 'shuffle_size'),
            ('epochs', 'valueChanged', 'epochs'),
            ('additional_epochs', 'valueChanged', 'additional_epochs'),                    
            ('batch_size', 'valueChanged', 'batch_size'),
            ('device_ID', 'valueChanged', 'device_id'),
            # 3. model evaluation tab page            
                        
            # 4. inference tab page           
            ('validation_size', 'valueChanged', 'validation_size'),
            ('inference_temperature', 'valueChanged', 'inference_temperature'),
            ('inference_mode', 'currentTextChanged', 'inference_mode')]        

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

        self.data_metrics = [('pixels_distribution', self.pixel_distribution_metric)]
        self.model_metrics = [('evaluation_report', self.get_evaluation_report),
                              ('BLEU_score', self.get_BLEU_score)]

    #--------------------------------------------------------------------------
    def _update_device(self):
        device = 'GPU' if self.use_GPU.isChecked() else 'CPU'
        self.config_manager.update_value('device', device)  

    #--------------------------------------------------------------------------
    def _set_states(self):         
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")        
        self.progress_bar.setValue(0)

    #--------------------------------------------------------------------------
    def get_current_pixmaps_and_key(self):
        for radio, (pixmap_key, idx_key) in self.pixmap_source_map.items():
            if radio.isChecked():
                return self.pixmaps[pixmap_key], idx_key
        return [], None 

    #--------------------------------------------------------------------------
    def _set_graphics(self):
        self.graphics = {}        
        view = self.main_win.findChild(QGraphicsView, 'canvas')
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem()
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        view.setRenderHint(QPainter.Antialiasing, True)
        view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        view.setRenderHint(QPainter.TextAntialiasing, True)
        self.graphics = {'view': view,
                         'scene': scene,
                         'pixmap_item': pixmap_item}            
                    
        self.pixmaps = {
            'train_images': [],         
            'inference_images': [],      
            'dataset_eval_images': [],  
            'model_eval_images': []}
        
        self.img_paths = {'train_images' : IMG_PATH,
                          'inference_images' : INFERENCE_INPUT_PATH}
              
        self.current_fig = {'train_images' : 0, 'inference_images' : 0,
                            'dataset_eval_images' : 0, 'model_eval_images' : 0}   
        
        self.pixmap_source_map = {
            self.data_plots_view: ("dataset_eval_images", "dataset_eval_images"),
            self.model_plots_view: ("model_eval_images", "model_eval_images"),
            self.inference_images_view: ("inference_images", "inference_images"),
            self.train_images_view: ("train_images", "train_images")} 
        
        self.text_view = {'train_images': self.dataset_handler.get_description_from_train_image,
                          'inference_images': self.dataset_handler.get_generated_report}

            
    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _start_worker(self, worker : Worker, on_finished, on_error, on_interrupted,
                      update_progress=True):
        if update_progress:       
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)        
        worker.signals.interrupted.connect(on_interrupted)
        self.threadpool.start(worker)
        self.worker_running = True

    #--------------------------------------------------------------------------
    def _send_message(self, message): 
        self.main_win.statusBar().showMessage(message)    

    # [SETUP]
    ###########################################################################
    def _setup_configuration(self, widget_defs):
        for cls, name, attr in widget_defs:
            w = self.main_win.findChild(cls, name)
            setattr(self, attr, w)
            self.widgets[attr] = w

    #--------------------------------------------------------------------------
    def _connect_signals(self, connections):
        for attr, signal, slot in connections:
            widget = self.widgets[attr]
            getattr(widget, signal).connect(slot)   
   
    # [SLOT]
    ###########################################################################
    # It's good practice to define methods that act as slots within the class
    # that manages the UI elements. These slots can then call methods on the
    # handler objects. Using @Slot decorator is optional but good practice
    #--------------------------------------------------------------------------
    Slot()
    def stop_running_worker(self):
        if self.worker is not None:
            self.worker.stop()       
        self._send_message("Interrupt requested. Waiting for threads to stop...")

    #--------------------------------------------------------------------------
    @Slot()
    def load_checkpoints(self):       
        checkpoints = self.model_handler.get_available_checkpoints()
        self.checkpoints_list.clear()
        if checkpoints:
            self.checkpoints_list.addItems(checkpoints)
            self.selected_checkpoint = checkpoints[0]
            self.checkpoints_list.setCurrentText(checkpoints[0])
        else:
            self.selected_checkpoint = None
            logger.warning("No checkpoints available")

    #--------------------------------------------------------------------------
    @Slot(str)
    def select_checkpoint(self, name: str):
        self.selected_checkpoint = name if name else None 

    #--------------------------------------------------------------------------
    @Slot()
    def _update_metrics(self):        
        self.selected_metrics['dataset'] = [
            name for name, box in self.data_metrics if box.isChecked()]
        self.selected_metrics['model'] = [
            name for name, box in self.model_metrics if box.isChecked()]

    #--------------------------------------------------------------------------
    # [GRAPHICS]
    #--------------------------------------------------------------------------
    @Slot(str)
    def _update_graphics_view(self):  
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            self.graphics['pixmap_item'].setPixmap(QPixmap())
            self.graphics['scene'].setSceneRect(0, 0, 0, 0)
            self.image_description.setPlainText('')
            return

        idx = self.current_fig.get(idx_key, 0)
        idx = min(idx, len(pixmaps) - 1)
        raw = pixmaps[idx]
        
        qpixmap = QPixmap(raw) if isinstance(raw, str) else raw
        view = self.graphics['view']
        pixmap_item = self.graphics['pixmap_item']
        scene = self.graphics['scene']
        view_size = view.viewport().size()
        scaled = qpixmap.scaled(view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_item.setPixmap(scaled)
        scene.setSceneRect(scaled.rect())
        # update the image description when necessary        
        self._update_image_descriptions(idx_key)

    #--------------------------------------------------------------------------
    def _update_image_descriptions(self, idx_key): 
        image_path = self.pixmaps[idx_key][self.current_fig[idx_key]]
        if isinstance(image_path, str):
            image_name = os.path.basename(image_path)
            description = self.text_view[idx_key](image_name)        
            self.image_description.setPlainText(description)
            return 
        
        placeholder = "No description available for this image."
        self.image_description.setPlainText(placeholder) 

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_previous_figure(self):             
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] > 0:
            self.current_fig[idx_key] -= 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self):
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] < len(pixmaps) - 1:
            self.current_fig[idx_key] += 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self):
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        self.pixmaps[idx_key].clear()
        self.current_fig[idx_key] = 0
        self._update_graphics_view()
        self.graphics['pixmap_item'].setPixmap(QPixmap())
        self.graphics['scene'].setSceneRect(0, 0, 0, 0)
        self.graphics['view'].viewport().update()
        self.image_description.setPlainText('') 

    #--------------------------------------------------------------------------    
    @Slot()
    def load_images(self):          
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if idx_key not in self.img_paths.keys():
            return
        
        self.pixmaps[idx_key].clear()
        self.configuration = self.config_manager.get_configuration() 
        self.dataset_handler = DatasetEvents(self.database, self.configuration)
        
        img_paths = self.dataset_handler.load_images_path(self.img_paths[idx_key])
        self.pixmaps[idx_key].extend(img_paths)
        self.current_fig[idx_key] = 0 
        self._update_graphics_view()    

    #--------------------------------------------------------------------------
    # [DATASET TAB]
    #--------------------------------------------------------------------------      
    @Slot()
    def run_dataset_evaluation_pipeline(self):  
        if not self.data_metrics:
            return 
        
        if self.worker_running:            
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.selected_metrics['dataset'])   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_dataset_evaluation_finished,
            on_error=self.on_evaluation_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def build_ML_dataset(self):          
        if self.worker_running:            
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.dataset_handler.build_ML_dataset)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_dataset_processing_finished,
            on_error=self.on_data_error,
            on_interrupted=self.on_task_interrupted)      

    #--------------------------------------------------------------------------
    # [TRAINING TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def train_from_scratch(self):
        if self.worker_running:            
            return 
                  
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)         
  
        # send message to status bar
        self._send_message("Training XREPORT Transformer model from scratch...")        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.model_handler.run_training_pipeline)                            
       
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self): 
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)   

        # send message to status bar
        self._send_message(f"Resume training from checkpoint {self.selected_checkpoint}")         
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.model_handler.resume_training_pipeline,
            self.selected_checkpoint)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)

    #--------------------------------------------------------------------------
    # [MODEL EVALUATION TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def run_model_evaluation_pipeline(self):  
        if self.worker_running:            
            return 

        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)    
        device = 'GPU' if self.use_GPU_evaluation.isChecked() else 'CPU'   
        # send message to status bar
        self._send_message(f"Evaluating {self.select_checkpoint} performances... ")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.validation_handler.run_model_evaluation_pipeline,
            self.selected_metrics['model'], self.selected_checkpoint, device)                
        
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)     

    #-------------------------------------------------------------------------- 
    @Slot()
    def get_checkpoints_summary(self):       
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)           
        # send message to status bar
        self._send_message("Generating checkpoints summary...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.validation_handler.get_checkpoints_summary) 

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    # [INFERENCE TAB]
    #--------------------------------------------------------------------------   
    @Slot()    
    def generate_reports_with_checkpoint(self):  
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)  
        device = 'GPU' if self.use_GPU_inference.isChecked() else 'CPU'
        # send message to status bar
        self._send_message(f"Generating reports from X-ray scans with {self.selected_checkpoint}") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.model_handler.run_inference_pipeline,
            self.selected_checkpoint,
            device)

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_inference_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)


    ###########################################################################
    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################
    def on_dataset_processing_finished(self, plots):         
        self.dataset_handler.handle_success(self.main_win, 'Dataset has been built successfully')
        self.worker_running = False

    #--------------------------------------------------------------------------   
    def on_dataset_evaluation_finished(self, plots):   
        key = 'dataset_eval_images'      
        if plots:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p) 
                 for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(self.main_win, 'Figures have been generated')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_train_finished(self, session):          
        self.model_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_model_evaluation_finished(self, plots):  
        key = 'model_eval_images'         
        if plots is not None:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p)
                for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(
            self.main_win, f'Model {self.selected_checkpoint} has been evaluated')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_inference_finished(self, session):         
        self.model_handler.handle_success(
            self.main_win, 'Inference call has been terminated')
        self.worker_running = False


    ###########################################################################   
    # [NEGATIVE OUTCOME HANDLERS]
    ###########################################################################     
    @Slot() 
    def on_data_error(self, err_tb):
        self.dataset_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False

    #--------------------------------------------------------------------------
    @Slot()     
    @Slot(tuple)
    def on_evaluation_error(self, err_tb):
        self.validation_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False  

    #--------------------------------------------------------------------------
    @Slot() 
    def on_model_error(self, err_tb):
        self.model_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False 

    #--------------------------------------------------------------------------
    def on_task_interrupted(self):         
        self.progress_bar.setValue(0)
        self._send_message('Current task has been interrupted by user') 
        logger.warning('Current task has been interrupted by user')
        self.worker_running = False        
        
          
         


        

    
    