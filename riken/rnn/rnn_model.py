import tensorflow as tf

from riken.prot_features import prot_features
from riken.prot_features.prot_features import chars

PSSM_DIM = None


# TODO: Add chemical properties like in Keras model or at least an interface that allows to choose features easily
class RnnModel:
    def __init__(self, lstm_size, n_classes, max_size, dropout_keep_p, optimizer, conv_n_filters,
                 input=None, pssm_input=None, labels=None):
        self.n_classes = n_classes
        self.max_size = max_size
        self.lstm_size = lstm_size
        self.dropout_keep_p = dropout_keep_p
        self.conv_n_filters = conv_n_filters

        self.cell_fn = tf.nn.rnn_cell.LSTMCell

        with tf.variable_scope('input'):
            if input is None and pssm_input is None and labels is None:
                self.input = tf.placeholder(tf.int32, shape=[None, self.max_size])
                self.pssm_input = tf.placeholder(tf.float32, shape=[None, self.max_size*PSSM_DIM])
                self.labels = tf.placeholder(tf.int32, shape=[None, ])
            elif (input is not None) and (pssm_input is not None) and (labels is not None):
                self.input = input
                self.pssm_input = pssm_input
                self.labels = labels
            else:
                raise ValueError

            one_hot = tf.one_hot(self.labels, self.n_classes)

            embed = tf.one_hot(self.input, depth=len(chars))
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

        with tf.variable_scope('core'):
            h = tf.layers.conv1d(h, filters=self.conv_n_filters, kernel_size=3, activation=tf.nn.relu, padding='same')
            h = tf.layers.dropout(h, rate=self.dropout_keep_p)

            # fw_lstm = self.cell_fn(num_units=self.lstm_size)
            # bw_lstm = self.cell_fn(num_units=self.lstm_size)
            # outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, h, dtype=tf.float32)
            # outputs = tf.concat(outputs, 2)

            outputs, state = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=128, direction='bidirectional')(h)

            # last_output = outputs[:, -1, :]
            attention = tf.layers.Dense(1)(outputs)
            attention = tf.squeeze(attention, axis=2)
            attention = tf.nn.softmax(attention, axis=1)
            attention = tf.tile(tf.expand_dims(attention, axis=2), multiples=[1, 1, 2*self.lstm_size])
            last_output = tf.multiply(outputs, attention)

            last_output = tf.reduce_sum(last_output, axis=1)
            # last_output = tf.tensordot(outputs, attention, axes=1)

        with tf.variable_scope('dense'):
            final = tf.layers.dense(last_output, self.n_classes, activation=None)
        print(final)
        self.logits = final
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=self.logits)
        self.gradient_loss = tf.gradients(self.loss, embed)[0]

        self.probabilities = tf.nn.softmax(logits=self.logits)
        print(self.labels)

        label_v = tf.cast(self.labels, dtype=tf.int32)
        pred_v = tf.argmax(self.probabilities, 1, output_type=tf.int32)
        acc, acc_op = tf.metrics.accuracy(label_v, pred_v)
        self.acc = acc, acc_op

        auc, auc_op = tf.metrics.auc(self.labels, self.probabilities[:, 1])
        # auc, auc_op = tf.metrics.auc(one_hot, self.probabilities)
        self.auc = auc, auc_op

        self.optimizer = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.init_op = tf.initialize_all_variables()

    @property
    def optimize(self):
        return self.optimizer


