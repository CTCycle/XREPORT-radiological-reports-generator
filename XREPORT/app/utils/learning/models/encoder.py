from __future__ import annotations

from typing import Any

from keras import layers, ops
from keras.saving import register_keras_serializable
from transformers import AutoImageProcessor, AutoModel

from XREPORT.app.utils.constants import ENCODERS_PATH


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

        self.model = AutoModel.from_pretrained(
            self.encoder_name, cache_dir=ENCODERS_PATH
        )
        if self.freeze_layers is True:
            for param in self.model.parameters():
                param.requires_grad = False

        self.processor = AutoImageProcessor.from_pretrained(
            self.encoder_name, cache_dir=ENCODERS_PATH, use_fast=True
        )
        self.dense = layers.Dense(self.embedding_dims)

    # call method
    # -------------------------------------------------------------------------
    def call(self, inputs: Any, **kwargs) -> Any:
        inputs = ops.transpose(inputs, axes=(0, 3, 1, 2))
        outputs = self.model(inputs, **kwargs)
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
