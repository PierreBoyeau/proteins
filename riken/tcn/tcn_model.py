import tensorflow as tf
from riken.protein_io import prot_features


class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)


def residual_block(input, dilatation, kernel_size, n_filters, name, dropout_rate, do1conv=True):
    conv = CausalConv1D(n_filters, kernel_size, dilation_rate=dilatation, activation=None)(input)
    conv = tf.layers.batch_normalization(conv)
    conv = tf.nn.relu(conv)
    conv = tf.nn.dropout(conv, keep_prob=dropout_rate)

    conv = CausalConv1D(n_filters, kernel_size, dilation_rate=dilatation, activation=None)(conv)
    conv = tf.layers.batch_normalization(conv)
    conv = tf.nn.relu(conv)
    conv = tf.nn.dropout(conv, keep_prob=dropout_rate)

    if do1conv:
        rescaled_input = tf.layers.conv1d(input, kernel_size=1, filters=n_filters)
    else:
        rescaled_input = input
    return rescaled_input + conv


class TCNModel:
    def __init__(self, n_classes, max_size, depth, kernel_size, n_filters, dropout_rate,
                 optimizer, input=None, pssm_input=None, labels=None):
        self.max_size = max_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes

        with tf.name_scope('transferable'):
            with tf.name_scope('input'):
                if input is None and pssm_input is None and labels is None:
                    self.input = tf.placeholder(tf.int32, shape=[None, self.max_size])
                    self.pssm_input = tf.placeholder(tf.float32, shape=[None, self.max_size * 42])
                    self.labels = tf.placeholder(tf.int32, shape=[None, ])
                elif (input is not None) and (pssm_input is not None):  # and (labels is not None):
                    self.input = input
                    self.pssm_input = pssm_input
                    self.labels = labels
                else:
                    print('input', input)
                    print('pssm input', pssm_input)
                    print('labels', labels)
                    raise ValueError
                embed = tf.one_hot(self.input, depth=len(prot_features.chars))
                static_feat_mat = tf.Variable(
                    # initial_value=prot_features.create_blosom_80_mat(),
                    initial_value=prot_features.create_overall_static_aa_mat(normalize=True),
                    dtype=tf.float32, trainable=False)

                inputs_replace_m1 = tf.nn.relu(self.input)  # Create indexes where -1 values (fill) are replaced by 0
                aa_static_feat = tf.nn.embedding_lookup(params=static_feat_mat, ids=inputs_replace_m1)

                # pssm_mat = tf.reshape(self.pssm_input, shape=[None, self.max_size, PSSM_DIM])
                pssm_mat = self.pssm_input

                # h = embed
                h = tf.concat([embed, aa_static_feat, pssm_mat], axis=2)
            with tf.name_scope('core'):
                h = self.res_block(h, dilatation=1, do1conv=True, name='resblock_1')
                for dilation_pow in range(1, self.depth):
                    h = self.res_block(h, dilatation=2**dilation_pow, name='resblock_{}'.format(dilation_pow),
                                       do1conv=True)

            with tf.name_scope('attention'):
                attention = tf.layers.Dense(1)(h)
                attention = tf.squeeze(attention, axis=2)
                attention = tf.nn.softmax(attention, axis=1)
                attention = tf.tile(tf.expand_dims(attention, axis=2), multiples=[1, 1, self.n_filters])
                last_output = tf.multiply(h, attention)
                last_output = tf.reduce_sum(last_output, axis=1)

        with tf.variable_scope('dense'):
            final = tf.layers.dense(last_output, self.n_classes, activation=None)
        self.logits = final
        self.probabilities = tf.nn.softmax(logits=self.logits)

        self.loss = None
        self.acc = None
        self.auc = None
        self.optimizer_fn = optimizer
        self.optimizer = None
        self.init_op = None

    def build(self):
        labels_one_hot = tf.one_hot(self.labels, self.n_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_one_hot, logits=self.logits)
        label_v = tf.cast(self.labels, dtype=tf.int32)
        pred_v = tf.argmax(self.probabilities, 1, output_type=tf.int32)
        acc, acc_op = tf.metrics.accuracy(label_v, pred_v)
        self.acc = acc, acc_op

        auc, auc_op = tf.metrics.auc(self.labels, self.probabilities[:, 1])
        self.auc = auc, auc_op
        self.optimizer = self.optimizer_fn.minimize(self.loss, global_step=tf.train.get_global_step())
        self.init_op = tf.initialize_all_variables()
        return

    def res_block(self, inputs, dilatation, do1conv, name):
        return residual_block(inputs, dilatation=dilatation, kernel_size=self.kernel_size, n_filters=self.n_filters,
                              name=name, dropout_rate=self.dropout_rate, do1conv=do1conv)

    @property
    def optimize(self):
        return self.optimizer
