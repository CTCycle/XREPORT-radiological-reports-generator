import json
import os

from XREPORT.app.constants import CONFIG_PATH


###############################################################################
class Configuration:
    def __init__(self) -> None:
        self.configuration = {
            # Dataset
            "seed": 42,
            "sample_size": 1.0,
            "validation_size": 0.2,
            "img_augmentation": False,
            "shuffle_dataset": True,
            "shuffle_size": 512,
            "max_report_size": 200,
            "tokenizer": "distilbert/distilbert-base-uncased",
            # Model
            "num_attention_heads": 3,
            "num_encoders": 2,
            "num_decoders": 2,
            "embedding_dimensions": 128,
            "freeze_img_encoder": True,
            "train_temperature": 1.0,
            "jit_compile": False,
            "jit_backend": "inductor",
            # Device
            "use_device_GPU": False,
            "device_id": 0,
            "use_mixed_precision": False,
            "num_workers": 0,
            # Training
            "split_seed": 76,
            "train_seed": 42,
            "epochs": 100,
            "additional_epochs": 10,
            "batch_size": 32,
            "use_tensorboard": False,
            "plot_training_metrics": True,
            "save_checkpoints": False,
            "checkpoints_frequency": 1,
            # Learning rate scheduler
            "use_scheduler": False,
            "post_warmup_LR": 0.001,
            "warmup_steps": 0,
            # Inference
            "inference_temperature": 1.0,
            "inference_mode": "greedy_search",
            # Validation
            "inference_batch_size": 20,
            "num_evaluation_samples": 10,
        }

    # -------------------------------------------------------------------------
    def get_configuration(self) -> dict[str, Any]:
        return self.configuration

    # -------------------------------------------------------------------------
    def update_value(self, key: str, value: Any) -> None:
        self.configuration[key] = value

    # -------------------------------------------------------------------------
    def save_configuration_to_json(self, name: str) -> None:
        full_path = os.path.join(CONFIG_PATH, f"{name}.json")
        with open(full_path, "w") as f:
            json.dump(self.configuration, f, indent=4)

    # -------------------------------------------------------------------------
    def load_configuration_from_json(self, name: str) -> None:
        full_path = os.path.join(CONFIG_PATH, name)
        with open(full_path) as f:
            self.configuration = json.load(f)
