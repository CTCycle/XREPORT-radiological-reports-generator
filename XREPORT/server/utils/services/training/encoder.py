from __future__ import annotations

from typing import Any

from keras import layers, ops
from keras.layers import TorchModuleWrapper
from keras.saving import register_keras_serializable
from transformers import AutoImageProcessor, AutoModel

from XREPORT.server.utils.constants import ENCODERS_PATH


# [PRETRAINED IMAGE ENCODER]
###############################################################################
@register_keras_serializable(package="Encoders", name="BeitXRayImageEncoder")
class BeitXRayImageEncoder(layers.Layer):
    def __init__(
        self, freeze_layers: bool = False, embedding_dims: int = 256, **kwargs
    ) -> None:
        super(BeitXRayImageEncoder, self).__init__(**kwargs)
        self.encoder_name = "microsoft/beit-base-patch16-224"
        self.freeze_layers = freeze_layers
        self.embedding_dims = embedding_dims

        # Load the pretrained BEiT model
        beit_model = AutoModel.from_pretrained(
            self.encoder_name, cache_dir=ENCODERS_PATH
        )
        if self.freeze_layers is True:
            for param in beit_model.parameters():
                param.requires_grad = False

        # Wrap with TorchModuleWrapper for Keras gradient tracking
        self.model = TorchModuleWrapper(beit_model)
        self.processor = AutoImageProcessor.from_pretrained(
            self.encoder_name, cache_dir=ENCODERS_PATH, use_fast=True
        )
        self.dense = layers.Dense(self.embedding_dims)

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        self.dense.build(input_shape[:-1] + (768,))  # BEiT outputs 768-dim
        super(BeitXRayImageEncoder, self).build(input_shape)

    # call method
    # -------------------------------------------------------------------------
    def call(self, inputs: Any, **kwargs) -> Any:
        inputs = ops.transpose(inputs, axes=(0, 3, 1, 2))
        # Do not pass kwargs to BeitModel - it doesn't accept arbitrary kwargs
        outputs = self.model(inputs)
        output = outputs.last_hidden_state
        output = self.dense(output)

        return output

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

