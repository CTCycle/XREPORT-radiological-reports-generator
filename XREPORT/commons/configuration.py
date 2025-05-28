###############################################################################
class Configuration:
    
    def __init__(self):
        self.configuration = {            
            'general_seed': 42,
            'split_seed': 76,
            'train_seed': 42,         

            # Dataset
            'sample_size': 1.0,
            'validation_size': 0.2,
            'img_augmentation': False,
            'shuffle_dataset': True,
            'shuffle_size': 1024,

            # Model 
            'initial_neurons': 64,
            'dropout_rate': 0.2,
            'jit_compile': False,
            'jit_backend': 'inductor',

            # Device
            'device': 'CPU',
            'device_id': 0,
            'use_mixed_precision': False,
            'num_workers': 0,

            # Training
            'train_sample_size': 1.0,
            'epochs': 100,
            'additional_epochs': 10,
            'batch_size': 32,
            'use_tensorboard': False,
            'plot_training_metrics' : True,
            'save_checkpoints': False,

            # Learning rate scheduler
            'use_scheduler' : False,
            'initial_lr': 0.001,
            'constant_steps': 40000,
            'decay_steps': 1000,
            'final_lr': 0.0001,

            # Validation
            'val_batch_size': 20,
            'num_evaluation_images': 6            
        }

    #--------------------------------------------------------------------------  
    def get_configuration(self):
        return self.configuration
    
    #--------------------------------------------------------------------------
    def update_value(self, key: str, value: bool):       
        self.configuration[key] = value