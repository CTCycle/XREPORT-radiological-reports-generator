from keras import layers, ops
from keras.config import floatx
from keras.utils import register_keras_serializable


# [POSITIONAL EMBEDDING]
###############################################################################
@register_keras_serializable(package="CustomLayers", name="PositionalEmbedding")
class PositionalEmbedding(layers.Layer):
    def __init__(
        self, vocabulary_size, embedding_dims, sequence_length, mask_zero=True, **kwargs
    ):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.mask_zero = mask_zero
        self.token_embeddings = layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=self.embedding_dims,
            mask_zero=mask_zero,
        )
        self.position_embeddings = layers.Embedding(
            input_dim=self.sequence_length, output_dim=self.embedding_dims
        )
        self.embedding_scale = ops.sqrt(ops.cast(self.embedding_dims, floatx()))

    # implement positional embedding through call method
    # -------------------------------------------------------------------------
    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=length, step=1)
        positions = ops.cast(positions, dtype=inputs.dtype)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens *= self.embedding_scale
        embedded_positions = self.position_embeddings(positions)
        full_embedding = embedded_tokens + embedded_positions

        if self.mask_zero:
            mask = ops.not_equal(inputs, 0)
            mask = ops.expand_dims(ops.cast(mask, floatx()), axis=-1)
            full_embedding *= mask

        return full_embedding

    # compute the mask for padded sequences
    # -------------------------------------------------------------------------
    def compute_mask(self, inputs, previous_mask = None):
        mask = ops.not_equal(inputs, 0)

        return mask

    # serialize layer for saving
    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(PositionalEmbedding, self).get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "sequence_length": self.sequence_length,
                "embedding_dims": self.embedding_dims,
                "mask_zero": self.mask_zero,
            }
        )
        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
