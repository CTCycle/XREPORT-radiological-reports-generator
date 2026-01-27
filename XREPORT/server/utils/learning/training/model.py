from __future__ import annotations

from typing import Any

from keras import Model, layers, optimizers
from torch import compile as torch_compile

from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.learning.training.encoder import BeitXRayImageEncoder
from XREPORT.server.utils.learning.training.layers import (
    PositionalEmbedding,
    SoftMaxClassifier,
    TransformerDecoder,
    TransformerEncoder,
)
from XREPORT.server.utils.learning.training.metrics import (
    MaskedAccuracy,
    MaskedSparseCategoricalCrossentropy,
)
from XREPORT.server.utils.learning.training.scheduler import WarmUpLRScheduler


# [XREPORT CAPTIONING MODEL]
###############################################################################
class XREPORTModel:
    def __init__(self, metadata: dict[str, Any], configuration: dict[str, Any]) -> None:
        self.seed = configuration.get("training_seed", 42)
        self.sequence_length = metadata.get("max_report_size", 200)
        self.vocabulary_size = metadata.get("vocabulary_size", 200)
        self.img_shape = (224, 224, 3)

        self.embedding_dims = configuration.get("embedding_dims", 256)
        self.num_heads = configuration.get("attention_heads", 8)
        self.num_encoders = configuration.get("num_encoders", 4)
        self.num_decoders = configuration.get("num_decoders", 4)
        self.freeze_img_encoder = configuration.get("freeze_img_encoder", False)
        self.jit_compile = configuration.get("jit_compile", False)
        self.jit_backend = configuration.get("jit_backend", "inductor")
        self.temperature = configuration.get("train_temp", 1.0)
        self.configuration = configuration
        self.metadata = metadata

        # initialize the image encoder and the transformers encoders and decoders
        self.img_input = layers.Input(shape=self.img_shape, name="image_input")
        self.seq_input = layers.Input(shape=(self.sequence_length,), name="seq_input")

        self.img_encoder = BeitXRayImageEncoder(
            self.freeze_img_encoder, self.embedding_dims
        )

        self.encoders = [
            TransformerEncoder(self.embedding_dims, self.num_heads, self.seed)
            for _ in range(self.num_encoders)
        ]
        self.decoders = [
            TransformerDecoder(self.embedding_dims, self.num_heads, self.seed)
            for _ in range(self.num_decoders)
        ]
        self.embeddings = PositionalEmbedding(
            self.vocabulary_size, self.embedding_dims, self.sequence_length
        )
        self.classifier = SoftMaxClassifier(
            1024, self.vocabulary_size, self.temperature
        )

    # -------------------------------------------------------------------------
    def compile_model(self, model: Model, model_summary: bool = True) -> Model:
        # Use target_LR (frontend naming) with fallback to post_warmup_LR (legacy naming)
        target_lr = self.configuration.get(
            "target_LR", self.configuration.get("post_warmup_LR", 0.0001)
        )
        lr_schedule: float | WarmUpLRScheduler = target_lr
        
        if self.configuration.get("use_scheduler", False):
            warmup_steps = self.configuration.get("warmup_steps", 100)
            lr_schedule = WarmUpLRScheduler(target_lr, warmup_steps)

        loss = MaskedSparseCategoricalCrossentropy()
        metric = [MaskedAccuracy()]
        opt = optimizers.AdamW(learning_rate=lr_schedule)  # type: ignore
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)  # type: ignore

        if model_summary:
            model.summary(expand_nested=True)
            
        if self.jit_compile:
            logger.info(f"JIT compiling model with backend: {self.jit_backend}")
            model = torch_compile(model, backend=self.jit_backend, mode="default")  # type: ignore

        return model

    # build model given the architecture
    # -------------------------------------------------------------------------
    def get_model(self, model_summary: bool = True) -> Model:
        # encode images and extract their features using the convolutional
        # image encoder or a selected pretrained model
        image_features = self.img_encoder(self.img_input)
        embeddings = self.embeddings(self.seq_input)
        padding_mask = self.embeddings.compute_mask(self.seq_input)

        encoder_output = image_features
        decoder_output = embeddings
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=False)
        for decoder in self.decoders:
            decoder_output = decoder(
                decoder_output, encoder_output, training=False, mask=padding_mask
            )

        # apply the softmax classifier layer
        output = self.classifier(decoder_output)

        # wrap the model and compile it with AdamW optimizer
        model = Model(inputs=[self.img_input, self.seq_input], outputs=output)
        model = self.compile_model(model, model_summary=model_summary)

        return model


def build_xreport_model(
    metadata: dict[str, Any], configuration: dict[str, Any]
) -> Model:
    """
    Build the XREPORT Transformer model for radiological report generation.
    
    Args:
        metadata: Dataset metadata containing vocabulary_size, max_report_size, etc.
        configuration: Training configuration with model hyperparameters.
    
    Returns:
        Compiled Keras model ready for training.
    """
    logger.info("Building XREPORT Transformer model")
    logger.info(f"  Vocabulary size: {metadata.get('vocabulary_size', 'unknown')}")
    logger.info(f"  Sequence length: {metadata.get('max_report_size', 'unknown')}")
    logger.info(f"  Embedding dims: {configuration.get('embedding_dims', 256)}")
    logger.info(f"  Attention heads: {configuration.get('attention_heads', 8)}")
    logger.info(f"  Encoders: {configuration.get('num_encoders', 4)}")
    logger.info(f"  Decoders: {configuration.get('num_decoders', 4)}")
    
    captioner = XREPORTModel(metadata, configuration)
    model = captioner.get_model(model_summary=True)
    
    logger.info("Model built successfully")
    return model
