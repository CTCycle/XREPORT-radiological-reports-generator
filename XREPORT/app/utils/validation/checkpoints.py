import os

import numpy as np
import pandas as pd
from keras import Model
from nltk.translate.bleu_score import corpus_bleu

from XREPORT.app.client.workers import check_thread_status, update_progress_callback
from XREPORT.app.constants import CHECKPOINT_PATH
from XREPORT.app.logger import logger
from XREPORT.app.utils.data.serializer import DataSerializer, ModelSerializer
from XREPORT.app.utils.learning.callbacks import LearningInterruptCallback
from XREPORT.app.utils.learning.inference.generator import TextGenerator


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:
    def __init__(self, configuration: dict, model: Model | None = None):
        self.serializer = DataSerializer()
        self.modser = ModelSerializer()
        self.model = model
        self.configuration = configuration

    # --------------------------------------------------------------------------
    def scan_checkpoint_folder(self) -> List[str]:
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                pretrained_model_path = os.path.join(entry.path, "saved_model.keras")
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)

        return model_paths

    # --------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs) -> pd.DataFrame:
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []
        for i, model_path in enumerate(model_paths):
            model = self.modser.load_checkpoint(model_path)
            configuration, metadata, history = self.modser.load_training_configuration(
                model_path
            )
            model_name = os.path.basename(model_path)
            precision = 16 if configuration.get("use_mixed_precision", np.nan) else 32
            has_scheduler = configuration.get("use_scheduler", False)
            scores = history.get("history", {})
            chkp_config = {
                "checkpoint": model_name,
                "sample_size": metadata.get("sample_size", np.nan),
                "validation_size": metadata.get("validation_size", np.nan),
                "seed": configuration.get("train_seed", np.nan),
                "precision": precision,
                "epochs": history.get("epochs", np.nan),
                "batch_size": configuration.get("batch_size", np.nan),
                "split_seed": metadata.get("split_seed", np.nan),
                "image_augmentation": configuration.get("img_augmentation", np.nan),
                "image_height": 224,
                "image_width": 224,
                "image_channels": 3,
                "jit_compile": configuration.get("jit_compile", np.nan),
                "has_tensorboard_logs": configuration.get("use_tensorboard", np.nan),
                "post_warmup_LR": configuration.get("post_warmup_LR", np.nan),
                "warmup_steps": configuration.get("warmup_steps", np.nan)
                if has_scheduler
                else np.nan,
                "temperature": configuration.get("train_temperature", np.nan),
                "tokenizer": configuration.get("tokenizer", np.nan),
                "max_report_size": metadata.get("max_report_size", np.nan),
                "attention_heads": configuration.get("attention_heads", np.nan),
                "n_encoders": configuration.get("num_encoders", np.nan),
                "n_decoders": configuration.get("num_decoders", np.nan),
                "embedding_dimensions": configuration.get(
                    "embedding_dimensions", np.nan
                ),
                "frozen_img_encoder": configuration.get("freeze_img_encoder", np.nan),
                "train_loss": scores.get("loss", [np.nan])[-1],
                "val_loss": scores.get("val_loss", [np.nan])[-1],
                "train_accuracy": scores.get("MaskedAccuracy", [np.nan])[-1],
                "val_accuracy": scores.get("val_MaskedAccuracy", [np.nan])[-1],
            }

            model_parameters.append(chkp_config)

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(model_paths), kwargs.get("progress_callback", None)
            )

        dataframe = pd.DataFrame(model_parameters)
        self.serializer.save_checkpoints_summary(dataframe)

        return dataframe

    # -------------------------------------------------------------------------
    def get_evaluation_report(self, model, validation_dataset, **kwargs):
        callbacks_list = [LearningInterruptCallback(kwargs.get("worker", None))]
        validation = model.evaluate(
            validation_dataset, verbose=1, callbacks=callbacks_list
        )
        logger.info(
            f"Sparse Categorical Entropy Loss {validation[0]:.3f} - Sparse Categorical Accuracy {validation[1]:.3f}"
        )


# [VALIDATION OF DATA]
###############################################################################
class EvaluateTextQuality:
    def __init__(
        self, model, configuration: dict, metadata: dict, num_samples: int | None = None
    ):
        self.model = model
        self.configuration = configuration
        self.metadata = metadata
        self.num_samples = num_samples

    # -------------------------------------------------------------------------
    def calculate_BLEU_score(self, validation_data: pd.DataFrame, **kwargs):
        max_report_size = self.metadata.get("max_report_size", 200)
        generator = TextGenerator(self.model, self.configuration, max_report_size)
        # tokenizer_config = generator.load_tokenizer_and_configuration()
        if self.num_samples is None:
            samples = validation_data.sample(n=self.num_samples, random_state=42)
        sampled_images = samples["path"].to_list()
        true_reports = dict(zip(samples["path"], samples["text"]))

        # Generate reports using greedy decoding
        generated_with_greedy = generator.generate_radiological_reports(
            sampled_images, method="greedy_search", worker=kwargs.get("worker", None)
        )

        references = []
        hypotheses = []

        # For each image, tokenize the corresponding ground-truth and generated reports.
        for i, image in enumerate(sampled_images):
            # Ensure that the image key exists in both the true reports and generated dictionary.
            if image in generated_with_greedy and image in true_reports:
                # Tokenize using simple split
                ref_tokens = true_reports[image].lower().split()
                cand_tokens = generated_with_greedy[image].lower().split()
                references.append([ref_tokens])
                hypotheses.append(cand_tokens)

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(sampled_images), kwargs.get("progress_callback", None)
            )

        # Calculate corpus BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        logger.info(
            f"BLEU score for {self.num_samples} validation samples: {bleu_score:.4f}"
        )

        return bleu_score
