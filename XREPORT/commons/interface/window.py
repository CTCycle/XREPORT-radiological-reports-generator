from XREPORT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

from functools import partial
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView)

from XREPORT.commons.configuration import Configuration
from XREPORT.commons.interface.events import ValidationEvents, TrainingEvents, InferenceEvents
from XREPORT.commons.interface.workers import Worker
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
        
        # Image data
        self.images_path = {'data': [], 'inference': []}
        self.image_pixmaps = []
        self.dataset_pixmaps = []
        self.inference_pixmaps = []
        self.model_eval_pixmaps = []

        # Canvas state
        self.canvas = ["imageCanvas", "modelEvalCanvas", "inferenceCanvas"]
        self.current_fig = {name: 0 for name in self.canvas}

        # Checkpoint & metrics state
        self.selected_checkpoint = None
        self.selected_metrics = {'dataset': [], 'model': []}       
          
        # initial settings
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()
    
        self.threadpool = QThreadPool.globalInstance()
        self.worker = None                

        # --- Create persistent handlers ---
        self.validation_handler = ValidationEvents(self.configuration)
        self.training_handler = TrainingEvents(self.configuration)
        self.inference_handler = InferenceEvents(self.configuration)

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([ 
            (QPushButton,'stopThread','stop_thread'),           
            # 1. dataset tab page
            (QCheckBox,'getStatsAnalysis','get_image_stats'),
            (QCheckBox,'getPixDist','get_pixels_dist'),
            (QPushButton,'getImgMetrics','get_img_metrics'),
            (QSpinBox,'seed','general_seed'),
            (QDoubleSpinBox,'sampleSize','sample_size'),
            (QPushButton,'loadImgDataset','load_source_images'),
            (QPushButton,'dataTabPreviousImg','data_tab_prev_img'),
            (QPushButton,'dataTabNextImg','data_tab_next_img'),
            (QPushButton,'dataTabClearImg','data_tab_clear_img'),
            (QRadioButton,'viewPlots','set_plot_view'),
            (QRadioButton,'viewImages','set_image_view'),
                      
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
            (QDoubleSpinBox,'initialLearningRate','initial_LR'),
            (QDoubleSpinBox,'targetLearningRate','target_LR'),            
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'), 
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),   
            (QComboBox,'backendJIT','jit_backend'),         
            (QSpinBox,'initialNeurons','initial_neurons'),
            (QDoubleSpinBox,'dropoutRate','dropout_rate'),                    
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),
            (QComboBox,'checkpointsList','checkpoints_list'),           
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            (QProgressBar,'trainingProgressBar','train_progress_bar'),
            # 3. model evaluation tab page
            (QPushButton,'evaluateModel','model_evaluation'),
            (QCheckBox,'runEvaluationGPU','use_GPU_evaluation'), 
            (QPushButton,'checkpointSummary','checkpoints_summary'),
            (QCheckBox,'evalReport','get_evaluation_report'), 
            (QCheckBox,'imgReconstruction','image_reconstruction'),      
            (QSpinBox,'numImages','num_evaluation_images'),
            (QPushButton,'evalTabPreviousImg','eval_tab_prev_img'),
            (QPushButton,'evalTabNextImg','eval_tab_next_img'),
            (QPushButton,'evalTabClearImg','eval_tab_clear_img'),    
            # 4. inference tab page  
            (QCheckBox,'runInferenceGPU','use_GPU_inference'),      
            (QPushButton,'encodeImages','encode_images'),
            (QPushButton,'loadInferenceImages','load_inference_images'),
            (QPushButton,'inferTabPreviousImg','infer_tab_prev_img'),
            (QPushButton,'inferTabNextImg','infer_tab_next_img'),
            (QPushButton,'inferTabClearImg','infer_tab_clear_img'),
            ])
        
        self._connect_signals([  
            ('checkpoints_list','currentTextChanged',self.select_checkpoint), 
            ('stop_thread','clicked',self.stop_running_worker),          
            # 1. dataset tab page
            ('get_image_stats','toggled',self._update_metrics),
            ('get_pixels_dist','toggled',self._update_metrics),
            ('get_img_metrics','clicked',self.run_dataset_evaluation_pipeline),
            ('load_source_images','clicked', self.load_image_dataset),
            ('data_tab_prev_img', 'clicked', lambda: self.show_previous_figure("imageCanvas")),
            ('data_tab_next_img', 'clicked', lambda: self.show_next_figure("imageCanvas")),
            ('data_tab_clear_img', 'clicked', lambda: self.clear_figures("imageCanvas")),
            ('set_plot_view', 'toggled', lambda: self._update_graphics_view("imageCanvas")),
            ('set_image_view', 'toggled', lambda: self._update_graphics_view("imageCanvas")),
            # 2. training tab page            
            ('refresh_checkpoints','clicked',self.load_checkpoints),
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),
            # 3. model evaluation tab page
            ('image_reconstruction','toggled',self._update_metrics),
            ('get_evaluation_report','toggled',self._update_metrics), 
            ('model_evaluation','clicked', self.run_model_evaluation_pipeline),
            ('checkpoints_summary','clicked',self.get_checkpoints_summary),                   
            ('eval_tab_prev_img','clicked', lambda: self.show_previous_figure("modelEvalCanvas")),     
            ('eval_tab_prev_img','clicked', lambda: self.show_previous_figure("modelEvalCanvas")),            
            ('eval_tab_next_img','clicked', lambda: self.show_next_figure("modelEvalCanvas")),
            ('eval_tab_clear_img','clicked', lambda: self.clear_figures("modelEvalCanvas")),
            # 4. inference tab page  
            ('encode_images','clicked',self.encode_images_with_checkpoint),
            ('load_inference_images','clicked', self.load_inference_dataset),
            ('infer_tab_prev_img','clicked', lambda: self.show_previous_figure("inferenceCanvas")),
            ('infer_tab_next_img','clicked', lambda: self.show_next_figure("inferenceCanvas")),
            ('infer_tab_clear_img','clicked', lambda: self.clear_figures("inferenceCanvas")),
        ]) 
        
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
            # 2. training tab page   
            ('img_augmentation', 'toggled', 'img_augmentation'),
            ('use_shuffle', 'toggled', 'shuffle_dataset'),
            ('num_workers', 'valueChanged', 'num_workers'),
            ('use_mixed_precision', 'toggled', 'mixed_precision'),
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
            ('initial_neurons', 'valueChanged', 'initial_neurons'),      
            ('dropout_rate', 'valueChanged', 'dropout_rate'),            
            ('batch_size', 'valueChanged', 'batch_size'),
            ('device_ID', 'valueChanged', 'device_id'),
            # 3. model evaluation tab page
            ('num_evaluation_images', 'valueChanged', 'num_evaluation_images'),            
            
            # 4. inference tab page           
            ('validation_size', 'valueChanged', 'validation_size')]        

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

        self.data_metrics = [('image_stats', self.get_image_stats), 
                             ('pixels_distribution', self.get_pixels_dist)]
        self.model_metrics = [('evaluation_report', self.get_evaluation_report),
                              ('image_reconstruction', self.image_reconstruction)]

    #--------------------------------------------------------------------------
    def _update_device(self):
        device = 'GPU' if self.use_GPU.isChecked() else 'CPU'
        self.config_manager.update_value('device', device)  

    #--------------------------------------------------------------------------
    def _set_states(self):         
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")        
        self.progress_bar.setValue(0) 

    #--------------------------------------------------------------------------
    def _set_graphics(self):
        self.graphics = {}        
        for canvas_name in self.canvas:
            view = self.main_win.findChild(QGraphicsView, canvas_name)
            scene = QGraphicsScene()
            pixmap_item = QGraphicsPixmapItem()
            pixmap_item.setTransformationMode(Qt.SmoothTransformation)
            scene.addItem(pixmap_item)
            view.setScene(scene)
            view.setRenderHint(QPainter.Antialiasing, True)
            view.setRenderHint(QPainter.SmoothPixmapTransform, True)
            view.setRenderHint(QPainter.TextAntialiasing, True)
            self.graphics[canvas_name] = {
                'view': view,
                'scene': scene,
                'pixmap_item': pixmap_item}
            
    #--------------------------------------------------------------------------
    def _load_single_pixmap(self, canvas_name, idx):    
        if canvas_name == "imageCanvas":
            self.image_pixmaps.clear()
            self.image_pixmaps.append(
                self.validation_handler.load_image_as_pixmap(self.images_path['data'][idx]))
        elif canvas_name == "inferenceCanvas":
            self.inference_pixmaps.clear()
            self.inference_pixmaps.append(
                self.validation_handler.load_image_as_pixmap(self.images_path['inference'][idx]))

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
    def select_image_source(self, canvas_name: str):
        if canvas_name == "imageCanvas":
            source = self.dataset_pixmaps if self.set_plot_view.isChecked() \
                     else self.image_pixmaps
        elif canvas_name == "inferenceCanvas":
            source = self.inference_pixmaps
        elif canvas_name == "modelEvalCanvas":
            source = self.model_eval_pixmaps 

        return source 
   
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
        checkpoints = self.training_handler.get_available_checkpoints()
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
        for name, box in self.data_metrics:
            if box.isChecked():
                if name not in self.selected_metrics['dataset']:
                    self.selected_metrics['dataset'].append(name)
            else:
                if name in self.selected_metrics['dataset']:
                    self.selected_metrics['dataset'].remove(name)

        for name, box in self.model_metrics:
            if box.isChecked():
                if name not in self.selected_metrics['model']:
                    self.selected_metrics['model'].append(name)
            else:
                if name in self.selected_metrics['model']:
                   self.selected_metrics['model'].remove(name)

    #--------------------------------------------------------------------------
    # [GRAPHICS]
    #--------------------------------------------------------------------------
    @Slot(str)
    def _update_graphics_view(self, canvas_name="imageCanvas"):         
        source = self.select_image_source(canvas_name)        
        if not source:
            return
        
        raw_pix = source[self.current_fig[canvas_name]] if len(source) > 1 else source[0]       
        view = self.graphics[canvas_name]['view']
        pixmap_item = self.graphics[canvas_name]['pixmap_item']
        scene = self.graphics[canvas_name]['scene']
        view_size = view.viewport().size()
        scaled = raw_pix.scaled(
            view_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        pixmap_item.setPixmap(scaled)
        scene.setSceneRect(scaled.rect())    

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_previous_figure(self, canvas_name="imageCanvas"):             
        if self.current_fig[canvas_name] == 0:
            return  

        self.current_fig[canvas_name] -= 1        
        # Decide what to show depending on canvas and view mode
        if canvas_name == "imageCanvas" and self.set_image_view.isChecked():
            self._load_single_pixmap(canvas_name, self.current_fig[canvas_name])
        elif canvas_name == "inferenceCanvas":
            self._load_single_pixmap(canvas_name, self.current_fig[canvas_name])        

        self._update_graphics_view(canvas_name)

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self, canvas_name="imageCanvas"):
        if canvas_name == "imageCanvas" and self.set_image_view.isChecked() \
            and self.current_fig[canvas_name] < len(self.images_path['data']):            
            self.current_fig[canvas_name] += 1 
            self._load_single_pixmap(canvas_name, self.current_fig[canvas_name])
        elif canvas_name == "inferenceCanvas" \
            and self.current_fig[canvas_name] < len(self.images_path['inference']): 
            self.current_fig[canvas_name] += 1 
            self._load_single_pixmap(canvas_name, self.current_fig[canvas_name])
        else:
            source = self.select_image_source(canvas_name)
            if self.current_fig[canvas_name] < len(source):             
                self.current_fig[canvas_name] += 1
        
        self._update_graphics_view(canvas_name)

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self, canvas_name="imageCanvas"):
        if canvas_name == "imageCanvas":
            self.image_pixmaps.clear()
            self.dataset_pixmaps.clear()
            self.images_path['data'].clear()
        elif canvas_name == "inferenceCanvas":
            self.inference_pixmaps.clear()
            self.images_path['inference'].clear()
        elif canvas_name == "modelEvalCanvas":
            self.model_eval_pixmaps.clear()

        self.current_fig[canvas_name] = 0
        # Blank out the graphics scene 
        gfx = self.graphics[canvas_name]
        # set the existing pixmap_item to an empty QPixmap
        gfx['pixmap_item'].setPixmap(QPixmap())
        # Shrink the scene rect so nothing is visible
        gfx['scene'].setSceneRect(0, 0, 0, 0)
        # Force an immediate repaint
        gfx['view'].viewport().update()

    #--------------------------------------------------------------------------
    # [DATASET TAB]
    #--------------------------------------------------------------------------    
    @Slot()
    def load_image_dataset(self):        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)
        self.images_path['data'].clear()             
        self.images_path['data'].extend(self.validation_handler.load_images_path(IMG_PATH)) 
        self.current_fig["imageCanvas"] = 0
        self.image_pixmaps.clear()        
        self.image_pixmaps.append(
            self.validation_handler.load_image_as_pixmap(
                self.images_path['data'][0])) 
        self._update_graphics_view("imageCanvas")            

    #--------------------------------------------------------------------------
    @Slot()
    def run_dataset_evaluation_pipeline(self):  
        if not self.data_metrics:
            return None
        
        self.get_img_metrics.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.selected_metrics['dataset'])   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_dataset_evaluation_finished,
            on_error=self.on_dataset_evaluation_error,
            on_interrupted=self.on_task_interrupted)       

    #--------------------------------------------------------------------------
    # [TRAINING TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def train_from_scratch(self):  
        self.start_training.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.training_handler = TrainingEvents(self.configuration)         
  
        # send message to status bar
        self._send_message("Training FEXT Autoencoder model from scratch...")        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.training_handler.run_training_pipeline)                            
       
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_train_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self):  
        self.resume_training.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.training_handler = TrainingEvents(self.configuration)   

        # send message to status bar
        self._send_message(f"Resume training from checkpoint {self.selected_checkpoint}")         
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.training_handler.resume_training_pipeline,
            self.selected_checkpoint)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_train_error,
            on_interrupted=self.on_task_interrupted)

    #--------------------------------------------------------------------------
    # [MODEL EVALUATION TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def run_model_evaluation_pipeline(self):  
        self.model_evaluation.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)    
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
            on_error=self.on_model_evaluation_error,
            on_interrupted=self.on_task_interrupted)     

    #-------------------------------------------------------------------------- 
    @Slot()
    def get_checkpoints_summary(self):
        self.checkpoints_summary.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)           
        # send message to status bar
        self._send_message("Generating checkpoints summary...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.validation_handler.get_checkpoints_summary) 

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_model_evaluation_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    # [INFERENCE TAB]
    #--------------------------------------------------------------------------
    @Slot()
    def load_inference_dataset(self):        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)        
        self.images_path['inference'].clear()        
        self.images_path['inference'].extend(
            self.validation_handler.load_images_path(INFERENCE_INPUT_PATH)) 
        self.current_fig["inferenceCanvas"] = 0
        self.inference_pixmaps.clear()        
        self.inference_pixmaps.append(
            self.validation_handler.load_image_as_pixmap(
                self.images_path['inference'][0])) 
        self._update_graphics_view("inferenceCanvas")            

    #--------------------------------------------------------------------------
    @Slot()    
    def encode_images_with_checkpoint(self):  
        self.encode_images.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.training_handler = InferenceEvents(self.configuration)  
        device = 'GPU' if self.use_GPU_inference.isChecked() else 'CPU'
        # send message to status bar
        self._send_message(f"Encoding images with {self.selected_checkpoint}") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.training_handler.run_inference_pipeline,
            self.selected_checkpoint,
            device)

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_inference_finished,
            on_error=self.on_inference_error,
            on_interrupted=self.on_task_interrupted)


    ###########################################################################
    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################       
    def on_dataset_evaluation_finished(self, plots):         
        if plots:
            self.dataset_pixmaps.extend(
                [self.validation_handler.convert_fig_to_qpixmap(p) 
                 for p in plots])
            
        self.current_fig["imageCanvas"] = 0
        self._update_graphics_view("imageCanvas")
        self.validation_handler.handle_success(self.main_win, 'Figures have been generated')
        self.get_img_metrics.setEnabled(True)

    #--------------------------------------------------------------------------
    def on_train_finished(self, session):          
        self.training_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.start_training.setEnabled(True) 
        self.resume_training.setEnabled(True)

    #--------------------------------------------------------------------------
    def on_model_evaluation_finished(self, plots):         
        if plots is not None:
            self.model_eval_pixmaps.extend(
                [self.validation_handler.convert_fig_to_qpixmap(p)
                for p in plots])
            
        self.current_fig["modelEvalCanvas"] = 0
        self._update_graphics_view("modelEvalCanvas")
        self.validation_handler.handle_success(
            self.main_win, f'Model {self.selected_checkpoint} has been evaluated')
        self.model_evaluation.setEnabled(True)

    #--------------------------------------------------------------------------
    def on_inference_finished(self, session):          
        self.training_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.encode_images.setEnabled(True)         

    ###########################################################################   
    # [NEGATIVE OUTCOME HANDLERS]
    ###########################################################################    
    @Slot(tuple)
    def on_dataset_evaluation_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.get_img_metrics.setEnabled(True)    

    #--------------------------------------------------------------------------
    @Slot() 
    def on_train_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.start_training.setEnabled(True) 
        self.resume_training.setEnabled(True) 

    #--------------------------------------------------------------------------
    @Slot() 
    def on_model_evaluation_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.get_img_metrics.setEnabled(True) 

    #--------------------------------------------------------------------------
    @Slot() 
    def on_inference_error(self, err_tb):
        self.inference_handler.handle_error(self.main_win, err_tb) 
        self.encode_images.setEnabled(True)

    #--------------------------------------------------------------------------
    def on_task_interrupted(self): 
        self.get_img_metrics.setEnabled(True) 
        self.encode_images.setEnabled(True)
        self.start_training.setEnabled(True) 
        self.resume_training.setEnabled(True) 
        self.model_evaluation.setEnabled(True)
        self.checkpoints_summary.setEnabled(True)
        self.progress_bar.setValue(0)
        self._send_message('Current task has been interrupted by user') 
        logger.warning('Current task has been interrupted by user')        
        
          
         


        

    
       

    
