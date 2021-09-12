from __future__ import absolute_import

from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.0,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        
        self.F_ = F_  # Number of output features
        self.attn_heads = attn_heads  # Number of attention heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel_1 = self.add_weight(shape=(F, self.F_),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      name='kernel1_{}'.format(head))
            kernel_2 = self.add_weight(shape=(F, self.F_),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      name='kernel2_{}'.format(head))
            self.kernels.append([kernel_1, kernel_2])

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       trainable=True,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               trainable=True,
                                               name='attn_kernel_{}'.format(head),)
            self.attn_kernels.append(attn_kernel)

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (B x N x F)
        A = inputs[1]  # Adjacency matrix (B x N x N)

        outputs = []
        Att = []
        for head in range(self.attn_heads):
            W1 = self.kernels[head][0]        # W1 shape: (F x F')
            W2 = self.kernels[head][1]        # W2 shape: (F x F')
            a = self.attn_kernels[head]       # Attention kernel a (F' x 1)

            # Compute HW
            HW1 = K.dot(X, W1)                                                 # (B x N x F')
            HW1_ext = tf.reshape(HW1, [1]+list(HW1.shape[1:])+[1])             # (B x N x F' x 1)
            HW2 = K.dot(X, W2)                                                 # (B x N x F')
            HW2_ext = tf.reshape(HW2, [1]+list(HW2.shape[1:])+[1])             # (B x N x F' x 1)
            HW2_permute = K.permute_dimensions(HW2_ext,(0,3,2,1))              # (B x 1 x F' x N)
            HW = HW1_ext + HW2_permute                                         # (B x N x F' x N)  via broadcasting
            HW = K.permute_dimensions(HW,(0,1,3,2))                            # (B x N x N x F')

            if self.use_bias:
                HW = K.bias_add(HW, self.biases[head])

            # Add nonlinearty
            f = LeakyReLU(alpha=0.2)(HW)                                   # (B x N x N x F')

            # Multiply by a
            att = K.dot(f, a)                                              # (B x N x N x 1)
            att = tf.squeeze(att, axis=-1)                                 # (B x N x N)
            
            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e15 * (1.0 - A)
            att += mask

            # Apply sigmoid to get attention coefficients
            att = K.sigmoid(att)                                           # (B x N x N)
            att_sum = K.sum(att, axis = -1, keepdims = True)
            att = att/(1 + att_sum)
            beta_promoter = 1/(1 + att_sum)
            
            Att.append(att)

            # Apply dropout to features and attention coefficients
            #dropout_attn = Dropout(self.dropout_rate)(att)                                              # (B x N x N)
            dropout_HW1 = Dropout(self.dropout_rate)(HW1)                                                # (B x N x F')
            dropout_HW2 = Dropout(self.dropout_rate)(HW2)                                                # (B x N x F')

            # Linear combination with neighbors' features
            node_features = beta_promoter * dropout_HW1 + K.batch_dot(att, dropout_HW2)                  # (B x N x F')

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)            # (B x N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # (B x N x F')

        output = self.activation(output)

        return output, Att

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'F_': self.F_,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'attn_kernel_initializer': self.attn_kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'attn_kernel_regularizer': self.attn_kernel_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'attn_kernel_constraint': self.attn_kernel_constraint,
        })
        return config  

