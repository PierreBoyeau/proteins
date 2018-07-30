import tensorflow as tf
import records_maker

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
        embed = tf.one_hot(self.input, depth=self.n_classes)

        blosom = tf.Variable(initial_value=records_maker.create_blosom_80_mat(),
                             dtype=tf.float32, trainable=False)
        inputs_replace_m1 = tf.nn.relu(self.input)  # Create indexes where -1 values (fill) are replaced by 0
        blosom_feat = tf.nn.embedding_lookup(params=blosom, ids=inputs_replace_m1)

        pssm_mat = tf.reshape(self.pssm_input, shape=[None, self.max_size, PSSM_DIM])

        # h = embed
        h = tf.concat([embed, blosom_feat, pssm_mat])

        h = tf.layers.conv1d(h, filters=100, kernel_size=3, activation=tf.nn.relu)
        h = tf.layers.dropout(h, rate=self.dropout_keep_p)
        cells = []
        # for sz in lstm_size_list:
        #     rnn = self.cell_fn(num_units=sz)
        #     # cells.append(rnn)
        #     # rnn = tf.nn.rnn_cell.DropoutWrapper(cell=rnn, output_keep_prob=dropout_keep_p)
        #     cells.append(rnn)
        # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        # outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
        #                                    inputs=h,
        #                                    dtype=tf.float32)
        # last_output = outputs[:, -1, :]

        # last_output = state[-1]
        # last_output = last_output.h

        fw_lstm = self.cell_fn(num_units=self.lstm_size)
        bw_lstm = self.cell_fn(num_units=self.lstm_size)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, h)
        outputs = tf.concat(outputs, 2)
        last_output = outputs[:, -1, :]

        final = tf.layers.dense(last_output, self.n_classes, activation=None)
        self.logits = final
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=self.logits)
        self.gradient_loss = tf.gradients(self.loss, embed)[0]

        self.probabilities = tf.nn.softmax(logits=self.logits)
        acc, acc_op = tf.metrics.accuracy(self.labels, tf.argmax(self.probabilities, 1))
        self.acc = acc

        self.optimizer = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.init_op = tf.initialize_all_variables()

    @property
    def optimize(self):
        return self.optimizer


