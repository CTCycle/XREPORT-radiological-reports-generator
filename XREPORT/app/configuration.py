###############################################################################
class Configuration:
    
    def __init__(self):
        self.configuration = {
            # Dataset
            'general_seed': 42,
            'sample_size': 1.0,
            'validation_size': 0.2,
            'img_augmentation': False,
            'shuffle_dataset': True,
            'shuffle_size': 512,
            'max_report_size': 200,
            'tokenizer': 'distilbert/distilbert-base-uncased',

            # Model 
            'num_attention_heads': 3,
            'num_encoders': 2,
            'num_decoders': 2,
            'embedding_dimensions' : 128,
            'freeze_img_encoder': True,
            'train_temperature': 1.0,
            'jit_compile': False,
            'jit_backend': 'inductor',

            # Device
            'use_device_GPU': False,
            'device_id': 0,
            'use_mixed_precision': False,
            'num_workers': 0,

            # Training
            'split_seed': 76,
            'train_seed': 42,            
            'epochs': 100,
            'additional_epochs': 10,
            'batch_size': 32,
            'use_tensorboard': False,
            'plot_training_metrics' : True,
            'save_checkpoints': False,

            # Learning rate scheduler
            'use_scheduler' : False,
            'post_warmup_LR': 0.001,
            'warmup_steps': 40000,           

            # Inference
            'inference_temperature': 1.0,
            'inference_mode': 'greedy_search',

            # Validation
            'inference_batch_size': 20,
            'num_evaluation_samples': 10
                        
        }

    #--------------------------------------------------------------------------  
    def get_configuration(self):
        return self.configuration
    
    #--------------------------------------------------------------------------
    def update_value(self, key: str, value: bool):       
        self.configuration[key] = value