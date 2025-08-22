from XREPORT.app.variables import EnvironmentVariables
EV = EnvironmentVariables()

import os
from functools import partial
from qt_material import apply_stylesheet
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt, QTimer
from PySide6.QtGui import QPainter, QPixmap, QAction
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView, QPlainTextEdit, QMessageBox, 
                               QDialog, QApplication)

from XREPORT.app.utils.data.database import database
from XREPORT.app.configuration import Configuration
from XREPORT.app.client.dialogs import LoadConfigDialog, SaveConfigDialog
from XREPORT.app.client.workers import ThreadWorker, ProcessWorker
from XREPORT.app.client.events import (GraphicsHandler, DatasetEvents, ValidationEvents, 
                                          ModelEvents)

from XREPORT.app.constants import IMG_PATH, INFERENCE_INPUT_PATH
from XREPORT.app.logger import logger

###############################################################################
def apply_style(app : QApplication):
    theme = 'dark_yellow'
    extra = {'density_scale': '-1'}
    apply_stylesheet(app, theme=f'{theme}.xml', extra=extra)

    # Make % text visible/centered for ALL progress bars
    app.setStyleSheet(app.styleSheet() + """
    QProgressBar {
        text-align: center;  /* align percentage to the center */
        color: black;        /* black text for yellow bar */
        font-weight: bold;   /* bold percentage */        
    }
    """)

    return app

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
    
        # set thread pool for the workers
        self.threadpool = QThreadPool.globalInstance()
        self.worker = None       

        # initialize database        
        database.initialize_database() 

        # --- Create persistent handlers ---
        self.graphic_handler = GraphicsHandler()
        self.dataset_handler = DatasetEvents(self.configuration)
        self.validation_handler = ValidationEvents(self.configuration)
        self.model_handler = ModelEvents(self.configuration)        

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([ 
            # actions
            (QAction, 'actionLoadConfig', 'load_configuration_action'),
            (QAction, 'actionSaveConfig', 'save_configuration_action'),
            (QAction, 'actionDeleteData', 'delete_data_action'),
            (QAction, 'actionExportData', 'export_data_action'),
            # out of tab widgets            
            (QProgressBar,'progressBar','progress_bar'),      
            (QPushButton,'stopThread','stop_thread'),
            # 1. dataset tab page 
            # data source group
            (QPushButton,'loadData','load_dataset'),
            (QSpinBox,'seed','seed'),
            (QDoubleSpinBox,'sampleSize','sample_size'),
            # dataset evaluation group                       
            (QCheckBox,'imgStats','image_statistics_metric'),
            (QCheckBox,'textStats','text_statistics_metric'),
            (QCheckBox,'pixDist','pixel_distribution_metric'),
            (QPushButton,'evaluateDataset','evaluate_dataset'),            
            # dataset processing group
            (QDoubleSpinBox,'validationSize','validation_size'),
            (QSpinBox,'splitSeed','split_seed'), 
            (QSpinBox,'maxReportSize','max_report_size'), 
            (QComboBox,'tokenizerList','tokenizer'),
            (QPushButton,'buildMLDataset','build_training_dataset'),
            # 2. training tab page
            # dataset settings group    
            (QCheckBox,'imgAugment','img_augmentation'),
            (QCheckBox,'setShuffle','use_shuffle'), 
            (QSpinBox,'shuffleSize','shuffle_size'),
            # training settings group
            (QCheckBox,'runTensorboard','use_tensorboard'),
            (QCheckBox,'realTimeHistory','real_time_history_callback'),
            (QCheckBox,'saveCheckpoints','save_checkpoints'),
            (QSpinBox,'saveCPFrequency','checkpoints_frequency'),            
            (QSpinBox,'numEpochs','epochs'),
            (QSpinBox,'batchSize','batch_size'),
            (QSpinBox,'trainSeed','train_seed'),  
            # model settings group
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),   
            (QComboBox,'backendJIT','jit_backend'), 
            (QCheckBox,'freezeImgEncoder','freeze_img_encoder'),          
            (QSpinBox,'attentionHeads','num_attention_heads'),
            (QSpinBox,'numEncoders','num_encoders'),                   
            (QSpinBox,'numDecoders','num_decoders'),
            (QSpinBox,'embeddingDims','embedding_dimensions'),
            (QCheckBox,'useScheduler','LR_scheduler'), 
            (QDoubleSpinBox,'postWarmLR','post_warmup_LR'),
            (QSpinBox,'warmupSteps','warmup_steps'),            
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'), 
            # session settings group  
            (QCheckBox,'deviceGPU','use_device_GPU'), 
            (QSpinBox,'deviceID','device_ID'),
            (QSpinBox,'numWorkers','num_workers'),
            (QDoubleSpinBox,'trainTemp','train_temperature'),
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),                     
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            # model inference and evaluation
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QComboBox,'checkpointsList','checkpoints_list'),
            (QSpinBox,'inferenceBatchSize','inference_batch_size'),     
            (QSpinBox,'evalSamples','num_evaluation_samples'), 
            (QPushButton,'evaluateModel','model_evaluation'),            
            (QPushButton,'checkpointSummary','checkpoints_summary'),
            (QCheckBox,'evalReport','get_evaluation_report'), 
            (QCheckBox,'getBLEUScore','get_BLEU_score'),            
            (QDoubleSpinBox,'inferenceTemp','inference_temperature'),
            (QComboBox,'inferenceMode','inference_mode'),                
            (QPushButton,'generateReports','generate_reports'),          
            # 3. Viewer tab
            (QPushButton,'loadImages','load_source_images'),
            (QPushButton,'previousImg','previous_image'),
            (QPushButton,'nextImg','next_image'),
            (QPushButton,'clearImg','clear_images'),
            (QRadioButton,'viewInferenceImages','inference_img_view'),
            (QRadioButton,'viewTrainImages','train_img_view'), 
            (QPlainTextEdit, 'description', 'image_description'),           
            ])
        
        self._connect_signals([ 
            # actions
            ('save_configuration_action', 'triggered', self.save_configuration),   
            ('load_configuration_action', 'triggered', self.load_configuration),
            ('delete_data_action', 'triggered', self.delete_all_data),   
            ('export_data_action', 'triggered', self.export_all_data),        
            # out of tab widgets    
            ('stop_thread','clicked',self.stop_running_worker),          
            # 1. dataset tab page       
            ('load_dataset','clicked',self.update_database_from_source),               
            ('image_statistics_metric','toggled',self._update_metrics),
            ('text_statistics_metric','toggled',self._update_metrics),
            ('pixel_distribution_metric','toggled',self._update_metrics),
            ('evaluate_dataset','clicked',self.run_dataset_evaluation_pipeline),
            ('build_training_dataset','clicked',self.run_dataset_builder),
            # 2. training tab page                                   
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),
            # model inference and evaluation
            ('checkpoints_list','currentTextChanged',self.select_checkpoint), 
            ('refresh_checkpoints','clicked',self.load_checkpoints),        
            ('get_evaluation_report','toggled',self._update_metrics), 
            ('get_BLEU_score','toggled',self._update_metrics),
            ('model_evaluation','clicked', self.run_model_evaluation_pipeline),
            ('checkpoints_summary','clicked',self.get_checkpoints_summary),              
            ('generate_reports','clicked',self.generate_reports_with_checkpoint),            
            # 3. viewer tab page 
            ('inference_img_view', 'toggled', self._update_graphics_view), 
            ('train_img_view', 'toggled', self._update_graphics_view), 
            ('load_source_images','clicked', self.load_images),
            ('previous_image', 'clicked', self.show_previous_figure),
            ('next_image', 'clicked', self.show_next_figure),
            ('clear_images', 'clicked', self.clear_figures),                        
        ]) 
        
        self._auto_connect_settings() 
               
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
            ('use_device_GPU', 'toggled', 'use_device_GPU'),
            # 1. dataset tab page
            # dataset evaluation group
            ('seed', 'valueChanged', 'seed'),
            ('sample_size', 'valueChanged', 'sample_size'),
            #  dataset processing group   
            ('max_report_size', 'valueChanged', 'max_report_size'),
            ('tokenizer', 'currentTextChanged', 'tokenizer'),
            ('split_seed', 'valueChanged', 'split_seed'), 
            # 2. training tab page
            # dataset settings group            
            ('img_augmentation', 'toggled', 'img_augmentation'),
            ('use_shuffle', 'toggled', 'shuffle_dataset'),
            ('shuffle_size', 'valueChanged', 'shuffle_size'),                       
            # device settings group
            ('device_ID', 'valueChanged', 'device_id'),
            ('num_workers', 'valueChanged', 'num_workers'),
            # training settings group
            ('use_tensorboard', 'toggled', 'use_tensorboard'),
            ('real_time_history_callback', 'toggled', 'real_time_history_callback'),
            ('save_checkpoints', 'toggled', 'save_checkpoints'),
            ('checkpoints_frequency', 'valueChanged', 'checkpoints_frequency'),
            ('epochs', 'valueChanged', 'epochs'),
            ('batch_size', 'valueChanged', 'batch_size'),
            ('train_seed', 'valueChanged', 'train_seed'),                       
            # RL scheduler settings group            
            ('LR_scheduler', 'toggled', 'use_LR_scheduler'),
            ('post_warmup_LR', 'valueChanged', 'post_warmup_LR'),
            ('warmup_steps', 'valueChanged', 'warmup_steps'),
            # model settings group
            ('use_mixed_precision', 'toggled', 'mixed_precision'),
            ('use_JIT_compiler', 'toggled', 'use_jit_compiler'),
            ('jit_backend', 'currentTextChanged', 'jit_backend'),
            ('freeze_img_encoder', 'toggled', 'freeze_img_encoder'),            
            ('num_attention_heads', 'valueChanged', 'num_attention_heads'),
            ('num_encoders', 'valueChanged', 'num_encoders'),
            ('num_decoders', 'valueChanged', 'num_decoders'),
            ('embedding_dimensions', 'valueChanged', 'embedding_dimensions'),
            ('train_temperature', 'valueChanged', 'train_temperature'),            
            # session settings group
            ('additional_epochs', 'valueChanged', 'additional_epochs'),
            # model inference and evaluation            
            ('inference_batch_size', 'valueChanged', 'inference_batch_size'),
            ('num_evaluation_samples', 'valueChanged', 'num_evaluation_samples'),                   
            ('inference_temperature', 'valueChanged', 'inference_temperature'),
            ('inference_mode', 'currentTextChanged', 'inference_mode')]   

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

        self.data_metrics = [('image_statistics', self.image_statistics_metric),
                             ('text_statistics', self.text_statistics_metric),
                             ('pixels_distribution', self.pixel_distribution_metric)]
        self.model_metrics = [('evaluation_report', self.get_evaluation_report),
                              ('BLEU_score', self.get_BLEU_score)]    

    #--------------------------------------------------------------------------
    def _set_states(self):         
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")        
        self.progress_bar.setValue(0)

    #--------------------------------------------------------------------------
    def get_current_pixmaps_key(self):
        for radio, idx_key in self.pixmap_sources.items():
            if radio.isChecked():
                return self.pixmaps[idx_key], idx_key
        return [], None 

    #--------------------------------------------------------------------------
    def _set_graphics(self):      
        view = self.main_win.findChild(QGraphicsView, 'canvas')
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem()
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        for hint in (QPainter.Antialiasing, QPainter.SmoothPixmapTransform, 
                     QPainter.TextAntialiasing):
            view.setRenderHint(hint, True)

        self.graphics = {'view': view, 'scene': scene, 'pixmap_item': pixmap_item}
        self.pixmaps = {k: [] for k in ('train_images', 'inference_images')}
        self.img_paths = {'train_images': IMG_PATH, 'inference_images': INFERENCE_INPUT_PATH}
        self.current_fig = {k: 0 for k in self.pixmaps}

        self.pixmap_sources = {self.inference_img_view: "inference_images",
                               self.train_img_view: "train_images"}   
            
    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _start_thread_worker(self, worker : ThreadWorker, on_finished, on_error, on_interrupted,
                      update_progress=True):
        if update_progress:       
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)        
        worker.signals.interrupted.connect(on_interrupted)
        self.threadpool.start(worker)

    #--------------------------------------------------------------------------
    def _start_process_worker(self, worker : ProcessWorker, on_finished, on_error, 
                              on_interrupted, update_progress=True):
        if update_progress:
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)

        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        worker.signals.interrupted.connect(on_interrupted)

        # Polling for results from the process queue
        self.process_worker_timer = QTimer()
        self.process_worker_timer.setInterval(100)  # Check every 100ms
        self.process_worker_timer.timeout.connect(worker.poll)
        worker._timer = self.process_worker_timer
        self.process_worker_timer.start()

        worker.start()

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

    #--------------------------------------------------------------------------
    def _set_widgets_from_configuration(self):
        cfg = self.config_manager.get_configuration()
        for attr, widget in self.widgets.items():
            if attr not in cfg:
                continue
            v = cfg[attr]
            # CheckBox
            if hasattr(widget, "setChecked") and isinstance(v, bool):
                widget.setChecked(v)
            # Numeric widgets (SpinBox/DoubleSpinBox)
            elif hasattr(widget, "setValue") and isinstance(v, (int, float)):
                widget.setValue(v)
            # PlainTextEdit/TextEdit
            elif hasattr(widget, "setPlainText") and isinstance(v, str):
                widget.setPlainText(v)
            # LineEdit (or any widget with setText)
            elif hasattr(widget, "setText") and isinstance(v, str):
                widget.setText(v) 
   
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
    def _update_metrics(self):        
        self.selected_metrics['dataset'] = [
            name for name, box in self.data_metrics if box.isChecked()]
        self.selected_metrics['model'] = [
            name for name, box in self.model_metrics if box.isChecked()]
        
    #--------------------------------------------------------------------------
    # [ACTIONS]
    #--------------------------------------------------------------------------
    @Slot()
    def save_configuration(self):
        dialog = SaveConfigDialog(self.main_win)
        if dialog.exec() == QDialog.Accepted:
            name = dialog.get_name()
            name = 'default_config' if not name else name            
            self.config_manager.save_configuration_to_json(name)
            self._send_message(f"Configuration [{name}] has been saved")

    #--------------------------------------------------------------------------
    @Slot()
    def load_configuration(self):
        dialog = LoadConfigDialog(self.main_win)
        if dialog.exec() == QDialog.Accepted:
            name = dialog.get_selected_config()
            self.config_manager.load_configuration_from_json(name)                
            self._set_widgets_from_configuration()
            self._send_message(f"Loaded configuration [{name}]")

    #--------------------------------------------------------------------------
    @Slot()
    def export_all_data(self):
        database.export_all_tables_as_csv()
        message = 'All data from database has been exported'
        logger.info(message)
        self._send_message(message)

    #--------------------------------------------------------------------------
    @Slot()
    def delete_all_data(self):      
        database.delete_all_data()        
        message = 'All data from database has been deleted'
        logger.info(message)
        self._send_message(message)

    #--------------------------------------------------------------------------
    # [GRAPHICS]
    #--------------------------------------------------------------------------
    @Slot(str)
    def _update_graphics_view(self):  
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            self.graphics['pixmap_item'].setPixmap(QPixmap())
            self.graphics['scene'].setSceneRect(0, 0, 0, 0)
            return

        idx = self.current_fig.get(idx_key, 0)
        idx = min(idx, len(pixmaps) - 1)
        raw = pixmaps[idx]
        
        qpixmap = QPixmap(raw) if isinstance(raw, str) else raw
        view = self.graphics['view']
        pixmap_item = self.graphics['pixmap_item']
        scene = self.graphics['scene']
        view_size = view.viewport().size()
        scaled = qpixmap.scaled(
            view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_item.setPixmap(scaled)
        scene.setSceneRect(scaled.rect())
        # update the image description when necessary        
        self._update_image_descriptions(idx_key)

    #--------------------------------------------------------------------------
    def _update_image_descriptions(self, idx_key): 
        image_path = self.pixmaps[idx_key][self.current_fig[idx_key]]
        if isinstance(image_path, str):
            image_name = os.path.basename(image_path)
            description = self.dataset_handler.get_description_from_image(image_name)        
            self.image_description.setPlainText(description)
            return 
        
        placeholder = "No description available for this image."
        self.image_description.setPlainText(placeholder) 

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_previous_figure(self):             
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] > 0:
            self.current_fig[idx_key] -= 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self):
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] < len(pixmaps) - 1:
            self.current_fig[idx_key] += 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self):
        pixmaps, idx_key = self.get_current_pixmaps_key()
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
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if idx_key not in self.img_paths.keys():
            return
        
        self.pixmaps[idx_key].clear()
        self.configuration = self.config_manager.get_configuration() 
        self.dataset_handler = DatasetEvents(self.configuration)
        
        img_paths = self.dataset_handler.load_img_path(self.img_paths[idx_key])
        self.pixmaps[idx_key].extend(img_paths)
        self.current_fig[idx_key] = 0 
        self._update_graphics_view()    

    #--------------------------------------------------------------------------
    # [DATASET TAB]
    #--------------------------------------------------------------------------
    @Slot()
    def update_database_from_source(self):
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return         
                
        # send message to status bar
        self._send_message("Updating database with source data...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(database.update_database_from_source)   

        # start worker and inject signals
        self._start_thread_worker(
            self.worker, on_finished=self.on_database_uploading_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def run_dataset_evaluation_pipeline(self):  
        if not self.selected_metrics['dataset']:
            return 
        
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.selected_metrics['dataset'])   

        # start worker and inject signals
        self._start_thread_worker(
            self.worker, on_finished=self.on_dataset_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def run_dataset_builder(self):          
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.dataset_handler = DatasetEvents(self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(self.dataset_handler.run_dataset_builder)   

        # start worker and inject signals
        self._start_thread_worker(
            self.worker, on_finished=self.on_dataset_processing_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)      

    #--------------------------------------------------------------------------
    # [TRAINING TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def train_from_scratch(self):
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
                  
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.configuration)         
  
        # send message to status bar
        self._send_message("Training XREPORT Transformer using a new model instance...")        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(self.model_handler.run_training_pipeline)                            
       
        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self): 
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 

        if not self.selected_checkpoint:
            logger.warning('No checkpoint selected for resuming training')
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.configuration)   

        # send message to status bar
        self._send_message(f"Resume training from checkpoint {self.selected_checkpoint}")         
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.model_handler.resume_training_pipeline,
            self.selected_checkpoint)   

        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)

    #--------------------------------------------------------------------------
    # [MODEL EVALUATION AND INFERENCE TAB]
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
    def run_model_evaluation_pipeline(self):  
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
        
        if not self.selected_checkpoint:
            logger.warning('No checkpoint selected for resuming training')
            return 

        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)       
        # send message to status bar
        self._send_message(f"Evaluating {self.selected_checkpoint} performances... ")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.validation_handler.run_model_evaluation_pipeline,
            self.selected_metrics['model'], self.selected_checkpoint)                
        
        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)     

    #-------------------------------------------------------------------------- 
    @Slot()
    def get_checkpoints_summary(self):       
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)           
        # send message to status bar
        self._send_message("Generating checkpoints summary...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(self.validation_handler.get_checkpoints_summary) 

        # start worker and inject signals
        self._start_thread_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------  
    @Slot()    
    def generate_reports_with_checkpoint(self):  
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
        if not self.selected_checkpoint:
            logger.warning('No checkpoint selected for resuming training')
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.configuration) 
       
        # send message to status bar
        self._send_message(f"Generating reports from X-ray scans with {self.selected_checkpoint}") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.model_handler.run_inference_pipeline,
            self.selected_checkpoint)

        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_inference_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)

    ###########################################################################
    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################
    def on_database_uploading_finished(self, source_data):   
        message = f'Database updated with current source data ({len(source_data)}) records'
        self._send_message(message)
        QMessageBox.information(self.main_win, "Database successfully updated", message)     
        self.worker = self.worker.cleanup()

    #-------------------------------------------------------------------------- 
    def on_dataset_processing_finished(self, plots):         
        self._send_message('Training dataset has been built successfully') 
        self.worker = self.worker.cleanup()       

    #--------------------------------------------------------------------------   
    def on_dataset_evaluation_finished(self, plots):   
        key = 'dataset_eval_images'      
        if plots:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p) 
                 for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self._send_message('Figures have been generated')
        self.worker = self.worker.cleanup()        

    #--------------------------------------------------------------------------
    def on_train_finished(self, session):          
        self._send_message('Training session is over. Model has been saved')  
        self.worker = self.worker.cleanup()      

    #--------------------------------------------------------------------------
    def on_model_evaluation_finished(self, plots):  
        key = 'model_eval_images'         
        if plots is not None:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p)
                for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self._send_message(f'Model {self.selected_checkpoint} has been evaluated')
        self.worker = self.worker.cleanup()
        
    #--------------------------------------------------------------------------
    def on_inference_finished(self, session):         
        self._send_message('Inference call has been terminated')  
        self.worker = self.worker.cleanup()     


    ###########################################################################   
    # [NEGATIVE OUTCOME HANDLERS]
    ########################################################################### 
    def on_error(self, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n{tb}")
        message = "An error occurred during the operation. Check the logs for details."
        QMessageBox.critical(self.main_win, 'Something went wrong!', message)
        self.progress_bar.setValue(0)      
        self.worker = self.worker.cleanup()  

    ###########################################################################   
    # [INTERRUPTION HANDLERS]
    ###########################################################################
    def on_task_interrupted(self):         
        self.progress_bar.setValue(0)
        self._send_message('Current task has been interrupted by user') 
        logger.warning('Current task has been interrupted by user') 
        self.worker = self.worker.cleanup()  



