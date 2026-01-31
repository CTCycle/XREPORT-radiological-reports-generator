from __future__ import annotations

from typing import Any

from keras import layers, ops
from keras.config import floatx
from keras.saving import register_keras_serializable


###############################################################################
@register_keras_serializable(package="CustomLayers", name="PositionalEmbedding")
class PositionalEmbedding(layers.Layer):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dims: int,
        sequence_length: int,
        mask_zero: bool = True,
        **kwargs,
    ) -> None:
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

    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        self.token_embeddings.build(input_shape)
        self.position_embeddings.build((input_shape[-1],))
        super(PositionalEmbedding, self).build(input_shape)

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
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

    # -------------------------------------------------------------------------
    def compute_mask(self, inputs: Any, previous_mask=None) -> Any:
        mask = ops.not_equal(inputs, 0)

        return mask

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

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[PositionalEmbedding], config: dict[str, Any]
    ) -> PositionalEmbedding:
        return cls(**config)


###############################################################################
@register_keras_serializable(package="CustomLayers", name="AddNorm")
class AddNorm(layers.Layer):
    def __init__(self, epsilon: float = 10e-5, **kwargs) -> None:
        super(AddNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization(epsilon=self.epsilon)
        self.supports_masking = True

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        super(AddNorm, self).build(input_shape)

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
        x1, x2 = inputs
        x_add = self.add([x1, x2])
        x_norm = self.layernorm(x_add)

        return x_norm

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(AddNorm, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls: type[AddNorm], config: dict[str, Any]) -> AddNorm:
        return cls(**config)


###############################################################################
@register_keras_serializable(package="CustomLayers", name="FeedForward")
class FeedForward(layers.Layer):
    def __init__(
        self, dense_units: int, dropout: float = 0.2, seed: int = 42, **kwargs
    ) -> None:
        super(FeedForward, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.dropout_rate = dropout
        self.dense1 = layers.Dense(
            dense_units, activation="relu", kernel_initializer="he_uniform"
        )
        self.dense2 = layers.Dense(
            dense_units, activation="relu", kernel_initializer="he_uniform"
        )
        self.dropout = layers.Dropout(rate=dropout, seed=seed)
        self.seed = seed

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        super(FeedForward, self).build(input_shape)

    # -------------------------------------------------------------------------
    def call(self, x: Any, training: bool | None = None) -> Any:
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dropout(x, training=training)
        return output

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(FeedForward, self).get_config()
        config.update(
            {
                "dense_units": self.dense_units,
                "dropout_rate": self.dropout_rate,
                "seed": self.seed,
            }
        )
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls: type[FeedForward], config: dict[str, Any]) -> FeedForward:
        return cls(**config)


###############################################################################
@register_keras_serializable(package="CustomLayers", name="SoftMaxClassifier")
class SoftMaxClassifier(layers.Layer):
    def __init__(
        self, dense_units: int, output_size: int, temperature: float = 1.0, **kwargs
    ) -> None:
        super(SoftMaxClassifier, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.output_size = output_size
        self.temperature = temperature
        self.dense1 = layers.Dense(dense_units, kernel_initializer="he_uniform")
        self.dense2 = layers.Dense(
            output_size, kernel_initializer="he_uniform", dtype=floatx()
        )
        self.supports_masking = True

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        super(SoftMaxClassifier, self).build(input_shape)

    # -------------------------------------------------------------------------
    def call(self, x: Any, training: bool | None = None) -> Any:
        from keras import activations

        layer = self.dense1(x)
        layer = activations.relu(layer)
        layer = self.dense2(layer)
        layer = layer / self.temperature
        output = activations.softmax(layer)

        return output

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(SoftMaxClassifier, self).get_config()
        config.update(
            {
                "dense_units": self.dense_units,
                "output_size": self.output_size,
                "temperature": self.temperature,
            }
        )
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[SoftMaxClassifier], config: dict[str, Any]
    ) -> SoftMaxClassifier:
        return cls(**config)


###############################################################################
@register_keras_serializable(package="Encoders", name="TransformerEncoder")
class TransformerEncoder(layers.Layer):
    def __init__(
        self, embedding_dims: int, num_heads: int, seed: int, **kwargs
    ) -> None:
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.seed = seed
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embedding_dims, seed=self.seed
        )
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed)
        # set mask supports to True but mask propagation is handled
        # through the attention layer call
        self.supports_masking: bool = True

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        super(TransformerEncoder, self).build(input_shape)

    # -------------------------------------------------------------------------
    def call(
        self, inputs: Any, mask: Any | None = None, training: bool | None = None
    ) -> Any:
        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        addnorm = self.addnorm1([inputs, attention_output])

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm, training=training)
        output = self.addnorm2([addnorm, ffn_out])

        return output

    # -------------------------------------------------------------------------
    def get_attention_scores(self) -> dict[str, Any]:
        return {}

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(TransformerEncoder, self).get_config()
        config.update(
            {
                "embedding_dims": self.embedding_dims,
                "num_heads": self.num_heads,
                "seed": self.seed,
            }
        )
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[TransformerEncoder], config: dict[str, Any]
    ) -> TransformerEncoder:
        return cls(**config)


###############################################################################
@register_keras_serializable(package="Decoders", name="TransformerDecoder")
class TransformerDecoder(layers.Layer):
    def __init__(
        self, embedding_dims: int, num_heads: int, seed: int, **kwargs
    ) -> None:
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.seed = seed
        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dims,
            dropout=0.2,
            seed=self.seed,
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dims,
            dropout=0.2,
            seed=self.seed,
        )
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed)
        # set mask supports to True but mask propagation is handled
        # through the attention layer call
        self.supports_masking: bool = True

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        super(TransformerDecoder, self).build(input_shape)

    # -------------------------------------------------------------------------
    def call(
        self,
        inputs: Any,
        encoder_outputs: Any,
        mask: Any | None = None,
        training: bool | None = None,
    ) -> Any:
        causal_mask = self.get_causal_attention_mask(inputs)
        combined_mask = causal_mask

        padding_mask: Any | None = None
        if mask is not None:
            padding_mask = ops.cast(ops.expand_dims(mask, axis=2), dtype="int32")
            combined_mask = ops.minimum(
                ops.cast(ops.expand_dims(mask, axis=1), dtype="int32"), causal_mask
            )

        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized
        self_masked_MHA = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        addnorm_out1 = self.addnorm1([inputs, self_masked_MHA])

        # cross attention using the encoder output as value and key and the output
        # of the addnorm layer as query. The output of this attention layer is then summed
        # to the inputs and normalized
        cross_MHA = self.cross_attention(
            query=addnorm_out1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        addnorm_out2 = self.addnorm2([addnorm_out1, cross_MHA])

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn = self.ffn1(addnorm_out2, training=training)
        logits = self.addnorm3([ffn, addnorm_out2])

        return logits

    # -------------------------------------------------------------------------
    def get_attention_scores(self) -> dict[str, Any]:
        return {}

    # -------------------------------------------------------------------------
    def get_causal_attention_mask(self, inputs: Any) -> Any:
        batch_size, sequence_length = ops.shape(inputs)[0], ops.shape(inputs)[1]
        i = ops.expand_dims(ops.arange(sequence_length), axis=1)
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype="int32")
        mask = ops.reshape(mask, (1, sequence_length, sequence_length))
        batch_mask = ops.tile(mask, (batch_size, 1, 1))

        return batch_mask

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(TransformerDecoder, self).get_config()
        config.update(
            {
                "embedding_dims": self.embedding_dims,
                "num_heads": self.num_heads,
                "seed": self.seed,
            }
        )
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[TransformerDecoder], config: dict[str, Any]
    ) -> TransformerDecoder:
        return cls(**config)
