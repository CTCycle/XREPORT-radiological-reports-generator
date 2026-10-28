from __future__ import annotations

from typing import Any

import torch
from keras import layers, ops
from keras.layers import TorchModuleWrapper
from keras.saving import register_keras_serializable
from transformers import AutoModel

from xreport_shared.common.path import ENCODERS_DIR
from xreport_shared.domain.checkpoint_exchange import ENCODER_REVISION as BEIT_REVISION


# [PRETRAINED IMAGE ENCODER]


###############################################################################
@register_keras_serializable(package="Encoders", name="BeitXRayImageEncoder")
class BeitXRayImageEncoder(layers.Layer):
    ENCODER_REVISION = BEIT_REVISION

    # -------------------------------------------------------------------------
    def __init__(
        self, freeze_layers: bool = False, embedding_dims: int = 256, **kwargs
    ) -> None:
        super(BeitXRayImageEncoder, self).__init__(**kwargs)
        self.encoder_name = "microsoft/beit-base-patch16-224"
        self.freeze_layers = freeze_layers
        self.embedding_dims = embedding_dims

        # Load the pretrained BEiT model
        beit_model = AutoModel.from_pretrained(
            self.encoder_name,
            revision=self.ENCODER_REVISION,
            cache_dir=ENCODERS_DIR,
            local_files_only=True,
        )
        if self.freeze_layers is True:
            for param in beit_model.parameters():
                param.requires_grad = False
        config = beit_model.config
        # CLS token plus one token per image patch.
        self.num_tokens = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size

        # Wrap with TorchModuleWrapper for Keras gradient tracking
        self.model = TorchModuleWrapper(beit_model)
        if self.freeze_layers is True:
            self.model.trainable = False
        self.dense = layers.Dense(self.embedding_dims)

    # build method for the custom layer

    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        # Built here because the meta-device shape trace skips call().
        self.dense.build((input_shape[0], self.num_tokens, self.hidden_size))
        super(BeitXRayImageEncoder, self).build(input_shape)

    # call method

    # -------------------------------------------------------------------------
    def call(self, inputs: Any, **kwargs) -> Any:
        # BEiT keeps float32 weights; a mixed policy would hand it float16 or
        # bfloat16 inputs, so cast back and let torch autocast pick the
        # reduced precision per op on CUDA.
        inputs = ops.cast(ops.transpose(inputs, axes=(0, 3, 1, 2)), "float32")
        compute_dtype = self.compute_dtype
        if inputs.device.type == "meta":
            # Keras traces shapes on the meta device first; BEiT's real weights
            # cannot run there, and the fallback trace would use the default
            # CUDA device even when training was placed on the CPU.
            return torch.empty(
                (inputs.shape[0], self.num_tokens, self.embedding_dims),
                dtype=getattr(torch, compute_dtype),
                device="meta",
            )
        # Do not pass kwargs to BeitModel - it doesn't accept arbitrary kwargs
        if compute_dtype != "float32" and inputs.device.type == "cuda":
            with torch.autocast("cuda", dtype=getattr(torch, compute_dtype)):
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        output = outputs.last_hidden_state
        output = self.dense(output)

        return output

    # -------------------------------------------------------------------------
    def compute_output_shape(self, input_shape: Any) -> tuple[Any, ...]:
        # Without this Keras infers the shape by running BEiT eagerly on a dummy
        # batch, on the default CUDA device even inside a CPU device scope.
        return (input_shape[0], self.num_tokens, self.embedding_dims)

    # serialize layer for saving

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(BeitXRayImageEncoder, self).get_config()
        config.update(
            {"freeze_layers": self.freeze_layers, "embedding_dims": self.embedding_dims}
        )

        return config

    # deserialization method

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[BeitXRayImageEncoder], config: dict[str, Any]
    ) -> BeitXRayImageEncoder:
        return cls(**config)
