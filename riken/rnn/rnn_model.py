import tensorflow as tf
import records_maker

class RnnModel:
    def __init__(self, lstm_size_list, n_classes, vocab_size, learning_rate, max_size, embed_dim,
                 dropout_keep_p, optimizer):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.lr = learning_rate
        self.max_size = max_size
        self.embed_dim = embed_dim

        self.cell_fn = tf.nn.rnn_cell.LSTMCell
        self.input = tf.placeholder(tf.int32, shape=[None, self.max_size])
        self.labels = tf.placeholder(tf.int32, shape=[None, ])
        # self.input = input
        # self.labels = labels

        one_hot = tf.one_hot(self.labels, self.n_classes)

        embed = tf.one_hot(self.input, depth=self.n_classes)

        blosom = tf.Variable(initial_value=records_maker.create_blosom_80_mat(),
                             dtype=tf.float32, trainable=False)
        inputs_replace_m1 = tf.nn.relu(self.input)  # Create indexes where -1 values (fill) are replaced by 0
        blosom_feat = tf.nn.embedding_lookup(params=blosom, ids=inputs_replace_m1)

        # h = embed
        h = tf.concat([embed, blosom_feat])

        h = tf.layers.conv1d(h, filters=100, kernel_size=3, activation=tf.nn.relu)

        cells = []
        for sz in lstm_size_list:
            rnn = self.cell_fn(num_units=sz)
            # cells.append(rnn)
            # rnn = tf.nn.rnn_cell.DropoutWrapper(cell=rnn, output_keep_prob=dropout_keep_p)
            cells.append(rnn)
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=h,
                                           dtype=tf.float32)

        # last_output = outputs[:, -1, :]
        last_output = state[-1]
        last_output = last_output.h

        final = tf.layers.dense(last_output, self.n_classes, activation=None)
        self.logits = final
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=self.logits)
        self.gradient_loss = tf.gradients(self.loss, embed)[0]

        acc, acc_op = tf.metrics.accuracy(self.labels, tf.argmax(self.logits, 1))
        self.acc = acc

        self.optimizer = optimizer.minimize(self.loss)
        self.init_op = tf.initialize_all_variables()

    @property
    def optimize(self):
        return self.optimizer


