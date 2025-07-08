import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes


class GATv2Conv(Conv):
    """Graph Attention v2 layer (Brody *et al.* ICLR 2022).

    Implements the decoupled formulation
        e_{ij} = a^T LeakyReLU( W_s x_i + W_t x_j )
        α_{ij} = softmax_j( e_{ij} )
        x'_i   = Σ_j α_{ij} · (W_t x_j)

    Parameters
    ----------
    channels: int
        Output dimensionality **per head**.
    attn_heads: int, default 1
        Number of attention heads.
    concat_heads: bool, default True
        If ``True`` concatenate the heads' outputs; otherwise average them.
    dropout_rate: float, default 0.0
        Dropout applied to the attention coefficients *during training*.
    return_attn_coef: bool, default False
        If ``True`` layer returns ``[output, attn]`` where ``attn`` has shape
        ``(batch, heads, N, N)``.
    activation: str | callable, default None
        Optional activation applied to the node embeddings.
    use_bias: bool, default True
        Add bias after aggregation.
    kernel_*, attn_kernel_*, bias_*: str | callable | None
        Initializer / regularizer / constraint for the corresponding weights.
    """

    def __init__(
        self,
        channels,
        attn_heads=1,
        concat_heads=True,
        dropout_rate=0.0,
        return_attn_coef=False,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        attn_kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        attn_kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        attn_kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(activation=activation, **kwargs)
        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = float(dropout_rate)
        self.return_attn_coef = return_attn_coef
        self.use_bias = use_bias

        # Serialise Keras objs
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    # ---------------------------------------------------------------------
    # Keras / Spektral internals
    # ---------------------------------------------------------------------
    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            node_features = input_shape[0][-1]
        else:
            node_features = input_shape[-1]

        # Two linear kernels (source, target) *per head*
        self.kernels_src, self.kernels_dst, self.attn_kernels = [], [], []
        for head in range(self.attn_heads):
            W_src = self.add_weight(
                name=f"kernel_src_{head}",
                shape=(node_features, self.channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            W_dst = self.add_weight(
                name=f"kernel_dst_{head}",
                shape=(node_features, self.channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            a = self.add_weight(
                name=f"attn_kernel_{head}",
                shape=(self.channels,),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
            )
            self.kernels_src.append(W_src)
            self.kernels_dst.append(W_dst)
            self.attn_kernels.append(a)

        if self.use_bias:
            self.biases = [
                self.add_weight(
                    name=f"bias_{h}",
                    shape=(self.channels,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )
                for h in range(self.attn_heads)
            ]
        else:
            self.biases = None
        super().build(input_shape)

    # ------------------------------------------------------------------
    def get_inputs(self, inputs):
        """Utility to unpack ``(X, A)`` vs ``X`` inputs and set mode."""
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            X, A = inputs[:2]
        else:
            X = inputs
            A = self.A  # Can be set externally for static graphs

        self.mode = modes.SINGLE if X.shape.ndims == 2 else modes.BATCH
        return X, A

    # ------------------------------------------------------------------
    def call(self, inputs, training=False, mask=None):  # noqa: D401
        X, A = self.get_inputs(inputs)

        outputs, attentions = [], []
        for h in range(self.attn_heads):
            Ws, Wt = self.kernels_src[h], self.kernels_dst[h]
            a = self.attn_kernels[h]
            b = self.biases[h] if self.use_bias else None

            Hs = tf.matmul(X, Ws)  # (.., N, C)
            Hd = tf.matmul(X, Wt)

            if self.mode == modes.SINGLE:
                out, alpha = self._atten_single(Hs, Hd, a, A, training)
            elif self.mode == modes.BATCH:
                out, alpha = self._atten_batch(Hs, Hd, a, A, training)
            else:
                raise NotImplementedError("Mixed/disjoint mode not implemented")

            if b is not None:
                out = out + b
            if self.activation is not None:
                out = self.activation(out)

            outputs.append(out)
            attentions.append(alpha)

        # Aggregate heads
        if self.concat_heads:
            output = tf.concat(outputs, axis=-1)
        else:
            output = tf.add_n(outputs) / self.attn_heads

        if self.return_attn_coef:
            attn = tf.stack(attentions, axis=1)  # (.., heads, N, N)
            return [output, attn]
        return output

    # ------------------------------------------------------------------
    # Attention helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_dropout(alpha, rate, training):
        if rate == 0.0:
            return alpha
        return tf.keras.backend.in_train_phase(
            tf.nn.dropout(alpha, rate=rate), alpha, training=training
        )

    def _atten_single(self, Hs, Hd, a, A, training):
        """Single‑graph attention."""
        # Broadcasting trick: Hi shape (N,1,C), Hj (1,N,C)
        Hi = tf.expand_dims(Hs, 1)
        Hj = tf.expand_dims(Hd, 0)
        e = tf.nn.leaky_relu(Hi + Hj, alpha=0.2)
        e = tf.tensordot(e, a, axes=[[2], [0]])  # (N,N)

        # Mask non‑edges; assume A is dense 0/1
        mask = tf.cast(A, tf.bool)
        e = tf.where(mask, e, tf.fill(tf.shape(e), -1e9))

        alpha = tf.nn.softmax(e, axis=-1)
        alpha = self._apply_dropout(alpha, self.dropout_rate, training)

        out = tf.matmul(alpha, Hd)  # (N,C)
        return out, alpha

    def _atten_batch(self, Hs, Hd, a, A, training):
        """Batch mode attention."""
        # Shapes: (B,N,1,C) & (B,1,N,C)
        Hi = tf.expand_dims(Hs, 2)
        Hj = tf.expand_dims(Hd, 1)
        e = tf.nn.leaky_relu(Hi + Hj, alpha=0.2)
        e = tf.tensordot(e, a, axes=[[3], [0]])  # (B,N,N)

        mask = tf.cast(A, tf.bool)
        e = tf.where(mask, e, tf.fill(tf.shape(e), -1e9))

        alpha = tf.nn.softmax(e, axis=-1)
        alpha = self._apply_dropout(alpha, self.dropout_rate, training)

        # einsum avoids explicit broadcast in matmul
        out = tf.einsum("bij,bjk->bik", alpha, Hd)
        return out, alpha

    # ------------------------------------------------------------------
    @property
    def config(self):
        config = super().config
        config.update(
            {
                "channels": self.channels,
                "attn_heads": self.attn_heads,
                "concat_heads": self.concat_heads,
                "dropout_rate": self.dropout_rate,
                "return_attn_coef": self.return_attn_coef,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "attn_kernel_initializer": initializers.serialize(
                    self.attn_kernel_initializer
                ),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "attn_kernel_regularizer": regularizers.serialize(
                    self.attn_kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "attn_kernel_constraint": constraints.serialize(
                    self.attn_kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config
