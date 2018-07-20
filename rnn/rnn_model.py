import tensorflow as tf


class RnnModel:
    def __init__(self, input, labels, lstm_size_list, n_classes, vocab_size, learning_rate, max_size, embed_dim,
                 dropout_keep_p, optimizer):
        self.n_classes = n_classes
        self.vocab_size= vocab_size
        self.lr = learning_rate
        self.max_size = max_size
        self.embed_dim = embed_dim

        self.cell_fn = tf.nn.rnn_cell.LSTMCell
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        # self.input = tf.placeholder(tf.int32, shape=[None, self.max_size])
        # self.labels = tf.placeholder(tf.int32, shape=[None,])
        self.input = input
        self.labels = labels

        one_hot = tf.one_hot(self.labels, self.n_classes)

        # embedding_mat = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -1, 1, dtype=tf.float32))
        # embed = tf.nn.embedding_lookup(embedding_mat, self.input)
        embed = tf.one_hot(self.input, depth=self.n_classes)
        h = embed
        # h = tf.layers.conv1d(embed, filters=100, kernel_size=3, activation=tf.nn.relu)

        cells = []
        for sz in lstm_size_list:
            rnn = self.cell_fn(num_units=sz)
            # cells.append(rnn)
            # rnn = tf.nn.rnn_cell.DropoutWrapper(cell=rnn, output_keep_prob=dropout_keep_p)
            cells.append(rnn)
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # multi_rnn_cell = self.cell_fn(num_units=128)
        # multi_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=multi_rnn_cell, output_keep_prob=dropout_keep_p)

        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=h,
                                           dtype=tf.float32)

        # last_output = outputs[:, -1, :]
        last_output = state[-1]
        last_output = last_output.h

        final = tf.layers.dense(last_output, self.n_classes, activation=None)
        # self.logits = tf.nn.softmax(final)
        self.logits = final
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=self.logits)
        self.gradient_loss = tf.gradients(self.loss, embed)[0]

        acc, acc_op = tf.metrics.accuracy(labels, tf.argmax(self.logits, 1))
        self.acc = acc

        self.optimizer = optimizer.minimize(self.loss)
        self.init_op = tf.initialize_all_variables()

    @property
    def optimize(self):
        return self.optimizer


