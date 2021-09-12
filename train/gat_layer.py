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
                 use_bias=False,
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
            kernel_self = self.add_weight(shape=(F, self.F_),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      name='kernel_self_{}'.format(head))
            kernel_neighs = self.add_weight(shape=(F, self.F_),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      name='kernel_neighs_{}'.format(head))
            self.kernels.append([kernel_self, kernel_neighs])

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
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               trainable=True,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 trainable=True,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (B x N x F)
        A = inputs[1]  # Adjacency matrix (B x N x N)

        outputs = []
        Att = []
        for head in range(self.attn_heads):
            kernel_self = self.kernels[head][0]         # W in the paper (F x F')
            kernel_neighs = self.kernels[head][1]
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features_self = K.dot(X, kernel_self)  # (B x N x F')
            features_neighs = K.dot(X, kernel_neighs)

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features_self, attention_kernel[0])      # (B x N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features_neighs, attention_kernel[1])  # (B x N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            attn_for_self_permute = K.permute_dimensions(attn_for_self,(1,0,2))       # (N x B x 1)
            attn_for_neighs_permute = K.permute_dimensions(attn_for_neighs,(1,0,2))   # (N x B x 1)
            att = attn_for_self_permute + K.transpose(attn_for_neighs_permute)        # (N x B x N) via broadcasting
            att = K.permute_dimensions(att,(1,0,2))                                   # (B x N x N)

            # Add nonlinearty
            att = LeakyReLU(alpha=0.2)(att)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e15 * (1.0 - A)
            att += mask

            # Apply sigmoid to get attention coefficients
            att = K.sigmoid(att)
            att_sum = K.sum(att, axis = -1, keepdims = True)
            att = att/(1 + att_sum)
            beta_promoter = 1/(1 + att_sum)

            Att.append(att)

            # Apply dropout to features and attention coefficients
            #dropout_attn = Dropout(self.dropout_rate)(att)                    # (B x N x N)
            dropout_feat_neigh = Dropout(self.dropout_rate)(features_neighs)   # (B x N x F')
            dropout_feat_self = Dropout(self.dropout_rate)(features_self)      # (B x N x F')

            # Linear combination with neighbors' features
            node_features = dropout_feat_self * beta_promoter + K.batch_dot(att, dropout_feat_neigh)  # (B x N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

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
