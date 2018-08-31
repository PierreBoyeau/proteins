import tensorflow as tf

from riken.protein_io import prot_features
from riken.protein_io.prot_features import chars

PSSM_DIM = None


# TODO: Add chemical properties like in Keras model or at least an interface that allows to choose features easily
class RnnModel:
    def __init__(self, lstm_size, n_classes, max_size, dropout_keep_p, optimizer, conv_n_filters,
                 two_lstm_layers=False, input=None, pssm_input=None, labels=None):
        self.n_classes = n_classes
        self.max_size = max_size
        self.lstm_size = lstm_size
        self.dropout_keep_p = dropout_keep_p
        self.conv_n_filters = conv_n_filters

        self.cell_fn = tf.nn.rnn_cell.LSTMCell
        # self.cell_fn = tf.nn.rnn_cell.GRUCell
        self.two_lstm_layers = two_lstm_layers

        with tf.name_scope('transferable'):
            with tf.name_scope('input'):
                if input is None and pssm_input is None and labels is None:
                    self.input = tf.placeholder(tf.int32, shape=[None, self.max_size])
                    self.pssm_input = tf.placeholder(tf.float32, shape=[None, self.max_size*PSSM_DIM])
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

            with tf.name_scope('core'):
                h = tf.layers.conv1d(h, filters=self.conv_n_filters, kernel_size=3, activation=tf.nn.relu, padding='same')
                h = tf.layers.dropout(h, rate=self.dropout_keep_p)

                # fw_lstm = self.cell_fn(num_units=self.lstm_size)
                # bw_lstm = self.cell_fn(num_units=self.lstm_size)
                # outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, h,
                # dtype=tf.float32, scope='BLSTM1')
                # outputs = tf.concat(outputs, 2)

                h = tf.transpose(h, perm=[1, 0, 2], name='inp_transpose_to_time_major')
                outputs, state = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                                num_units=self.lstm_size,
                                                                input_mode='linear_input',
                                                                direction='bidirectional',
                                                                dropout=0.25)(h)
                outputs = tf.transpose(outputs, perm=[1, 0, 2], name='out_transpose_to_time_major')

                # if self.two_lstm_layers:
                #     fw_lstm_2 = self.cell_fn(num_units=self.lstm_size,)
                #     bw_lstm_2 = self.cell_fn(num_units=self.lstm_size,)
                #     outputs_2, state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_2, bw_lstm_2,
                #                                                        outputs,
                #                                                        dtype=tf.float32,
                #                                                        scope='BLSTM2')
                #     outputs = tf.concat(outputs_2, 2)

                outputs = tf.layers.dropout(outputs, rate=self.dropout_keep_p)
                attention = tf.layers.Dense(1)(outputs)
                attention = tf.squeeze(attention, axis=2)
                attention = tf.nn.softmax(attention, axis=1)
                attention = tf.tile(tf.expand_dims(attention, axis=2), multiples=[1, 1, 2*self.lstm_size])
                last_output = tf.multiply(outputs, attention)
                last_output = tf.reduce_sum(last_output, axis=1)
                self.attention_output = last_output

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
        self.optimizer = self.optimizer_fn.minimize(self.loss,
                                                    global_step=tf.train.get_global_step())
        self.init_op = tf.initialize_all_variables()
        return

    @property
    def optimize(self):
        return self.optimizer


class RnnDecoder:
    def __init__(self, encoder_input, n_hidden=10, lstm_size=25, max_size=500):
        self.n_hidden = n_hidden
        self.encoder_input = encoder_input
        self.cell_fn = tf.nn.rnn_cell.LSTMCell
        # self.cell_fn = tf.nn.rnn_cell.GRUCell
        self.cell_fn = tf.contrib.cudnn_rnn.CudnnLSTM
        self.lstm_size = lstm_size
        self.max_size = max_size

        self.means = tf.layers.dense(self.encoder_input, units=self.n_hidden)
        self.log_sgm = tf.layers.dense(self.encoder_input, units=self.n_hidden)
        eps = tf.random_normal(shape=tf.shape(self.means), mean=0.1, stddev=1.0, dtype=tf.float32)

        # VAE
        # self.h = self.means + tf.multiply(eps, tf.exp(self.log_sgm))
        # AE CLASSIQUE
        self.h = self.means

        h = tf.expand_dims(self.h, axis=1)
        paddings = [[0, 0], [0, self.max_size-1], [0, 0]]
        h = tf.pad(h, paddings)
        print('H PADDED', h)
        # h = tf.tile(h, multiples=[1, self.max_size, 1])

        # fw_lstm = self.cell_fn(num_units=self.lstm_size)
        # bw_lstm = self.cell_fn(num_units=self.lstm_size)
        # outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, h, dtype=tf.float32,
        #                                                  scope='BLSTM_Decoder')
        # outputs = tf.concat(outputs, 2)

        h = tf.transpose(h, perm=[1, 0, 2], name='inp_transpose_to_time_major')
        outputs, state = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                        num_units=self.lstm_size,
                                                        input_mode='linear_input',
                                                        direction='bidirectional',
                                                        dropout=0.25)(h)
        outputs = tf.transpose(outputs, perm=[1, 0, 2], name='out_transpose_to_time_major')

        # IMPORTANT: Here I suppose that there are in total
        self.logits = tf.layers.dense(outputs, units=len(chars)+1)
        self.probabilities = tf.nn.softmax(self.logits, axis=2)
        self.predictions = tf.argmax(self.probabilities, axis=2)  # 0 to char_max+1
