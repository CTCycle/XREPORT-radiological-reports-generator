from __future__ import annotations

import math
from typing import Any

from keras import Model, layers, optimizers
from torch import compile as torch_compile

from server.common.utils.logger import logger
from server.domain.model_architecture import ModelArchitecture
from server.models.training.architecture import IMAGE_SHAPE, stack_kwargs
from server.models.training.encoder import BeitXRayImageEncoder
from server.models.training.layers import (
    PositionalEmbedding,
    SoftMaxClassifier,
    TransformerDecoder,
    TransformerEncoder,
)
from server.models.training.metrics import (
    MaskedAccuracy,
    MaskedSparseCategoricalCrossentropy,
)
from server.models.training.scheduler import WarmUpLRScheduler

###############################################################################
class XREPORTModel:

    # -------------------------------------------------------------------------
    def __init__(self, metadata: dict[str, Any], configuration: dict[str, Any]) -> None:
        self.seed = configuration.get("training_seed", 42)
        self.sequence_length = metadata.get("max_report_size", 200)
        self.vocabulary_size = metadata.get("vocabulary_size", 200)
        self.img_shape = IMAGE_SHAPE

        self.architecture = ModelArchitecture.model_validate(
            configuration["architecture"]
        )
        self.jit_compile = configuration.get("jit_compile", False)
        self.jit_backend = configuration.get("jit_backend", "inductor")
        self.configuration = configuration
        self.metadata = metadata

        architecture = self.architecture
        self.img_input = layers.Input(shape=self.img_shape, name="image_input")
        self.seq_input = layers.Input(shape=(self.sequence_length,), name="seq_input")

        # BEiT is the only registered image encoder (see ENCODER_REGISTRY)
        self.img_encoder = BeitXRayImageEncoder(
            architecture.freeze_image_encoder, architecture.embedding_dims
        )

        self.encoders = [
            TransformerEncoder(seed=self.seed, **stack_kwargs(architecture, "encoder"))
            for _ in range(architecture.num_encoders)
        ]
        self.decoders = [
            TransformerDecoder(seed=self.seed, **stack_kwargs(architecture, "decoder"))
            for _ in range(architecture.num_decoders)
        ]
        self.embeddings = PositionalEmbedding(
            self.vocabulary_size, architecture.embedding_dims, self.sequence_length
        )
        self.classifier = SoftMaxClassifier(
            architecture.classifier_units,
            self.vocabulary_size,
            architecture.temperature,
            architecture.classifier_activation,
        )

    # -------------------------------------------------------------------------
    def compile_model(self, model: Model, model_summary: bool = True) -> Model:
        target_lr = self.configuration.get("target_LR", 0.0001)
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

    # -------------------------------------------------------------------------
    def get_model(self, model_summary: bool = True) -> Model:
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

        output = self.classifier(decoder_output)

        model = Model(inputs=[self.img_input, self.seq_input], outputs=output)
        model = self.compile_model(model, model_summary=model_summary)
        if not model.built:
            model.build(
                [
                    (None, *self.img_shape),
                    (None, self.sequence_length),
                ]
            )

        return model

###############################################################################
def build_xreport_model(
    metadata: dict[str, Any], configuration: dict[str, Any]
) -> Model:
    logger.info("Building XREPORT Transformer model")
    logger.info(f"  Vocabulary size: {metadata.get('vocabulary_size', 'unknown')}")
    logger.info(f"  Sequence length: {metadata.get('max_report_size', 'unknown')}")
    logger.info(f"  Architecture: {configuration.get('architecture')}")

    captioner = XREPORTModel(metadata, configuration)
    model = captioner.get_model(model_summary=True)

    logger.info(
        "Model built successfully (%d parameters, %d trainable)",
        *count_model_parameters(model).values(),
    )
    return model

# -----------------------------------------------------------------------------
def count_model_parameters(model: Model) -> dict[str, int]:
    """Parameter counts of a built model, as stored with its checkpoint."""
    trainable = sum(int(math.prod(weight.shape)) for weight in model.trainable_weights)
    return {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": trainable,
    }
